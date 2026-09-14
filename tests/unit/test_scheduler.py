"""The scheduler, exercised end to end with fake agents.

This is Phase 1's exit criterion made concrete: "the full graph executes end
to end with fake agents; killing the process mid-run and resuming reproduces
identical state." Every scenario here is a distinct graph shape, kept small
and separate so a failure points at one mechanism, not a tangle of them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from ases.kernel.checkpoint import CheckpointStore, resume
from ases.kernel.events import EventType
from ases.kernel.gates import ApprovalDecision, ApprovalProvider, BudgetEntryGate, BudgetLimits
from ases.kernel.graph import Edge, EdgeCondition, JoinPolicy, NodeKind, NodeSpec, WorkflowGraph
from ases.kernel.scheduler import ConfigurationError, NodeExecutionOutcome, NodeExecutor, Scheduler
from ases.kernel.state import NodeStatus, RunState, RunStatus, fold
from ases.kernel.store.jsonl import JsonlEventStore
from tests.unit.fakes import (
    FixedExecutor,
    ScriptedApprovals,
    SequenceExecutor,
    SlowExecutor,
    fail,
    ok,
)

GENEROUS = BudgetLimits(max_tokens=10_000_000, max_usd=1_000.0, max_wallclock_seconds=3600)


def agent(node_id: str, **kw: object) -> NodeSpec:
    return NodeSpec(id=node_id, kind=NodeKind.AGENT, handler=node_id, **kw)  # type: ignore[arg-type]


def terminal(node_id: str = "done") -> NodeSpec:
    return NodeSpec(id=node_id, kind=NodeKind.TERMINAL)


def make_scheduler(
    graph: WorkflowGraph,
    store: JsonlEventStore,
    executors: dict[str, NodeExecutor],
    *,
    approvals: ApprovalProvider | None = None,
    limits: BudgetLimits = GENEROUS,
    checkpoints: CheckpointStore | None = None,
) -> Scheduler:
    return Scheduler(
        graph,
        store,
        executors,
        entry_gate=BudgetEntryGate(limits),
        approvals=approvals,
        checkpoints=checkpoints,
    )


# --- 1. linear success, with a granted approval gate ------------------------


async def test_linear_run_with_granted_approval_completes(
    store: JsonlEventStore, run_id: UUID
) -> None:
    graph = WorkflowGraph(
        name="linear",
        entry=("req",),
        nodes=(
            agent("req"),
            NodeSpec(id="gate1", kind=NodeKind.GATE, requires_approval=True),
            terminal(),
        ),
        edges=(Edge(source="req", target="gate1"), Edge(source="gate1", target="done")),
    )
    executors = {"req": FixedExecutor(ok(kind="RequirementSpec", text="build a thing"))}
    approvals = ScriptedApprovals({"gate1": [ApprovalDecision(granted=True, actor="alice")]})

    state = await make_scheduler(graph, store, executors, approvals=approvals).run(run_id)

    assert state.status is RunStatus.COMPLETED
    assert state.nodes["req"].status is NodeStatus.SUCCEEDED
    assert state.nodes["gate1"].status is NodeStatus.SUCCEEDED
    assert state.nodes["done"].status is NodeStatus.SUCCEEDED
    assert state.approvals["gate1"].granted
    assert len(state.artifacts) == 1


async def test_artifact_content_is_recorded_alongside_its_hash(
    store: JsonlEventStore, run_id: UUID
) -> None:
    """Phase 4 prerequisite: a downstream node (or a human at an approval
    gate) must be able to read what a producing node actually made, not just
    its hash. `ARTIFACT_PRODUCED` now carries the full payload, and the fold
    stores it in `RunState.artifact_content` - this is what
    `context.retriever.ContextRetriever.fetch` reads from."""
    graph = WorkflowGraph(
        name="content",
        entry=("req",),
        nodes=(agent("req"), terminal()),
        edges=(Edge(source="req", target="done"),),
    )
    executors = {
        "req": FixedExecutor(ok(kind="RequirementSpec", summary="s", source_text="raw text"))
    }

    state = await make_scheduler(graph, store, executors).run(run_id)

    artifact_hash = state.nodes["req"].produced[0]
    assert state.artifact_content[artifact_hash] == {"summary": "s", "source_text": "raw text"}


async def test_config_error_for_unregistered_handler(store: JsonlEventStore, run_id: UUID) -> None:
    """A node naming a handler nobody registered is an authoring mistake -
    it must raise, not silently hang or skip the node."""
    graph = WorkflowGraph(
        name="g",
        entry=("req",),
        nodes=(agent("req"), terminal()),
        edges=(Edge(source="req", target="done"),),
    )
    with pytest.raises(ConfigurationError, match="req"):
        await make_scheduler(graph, store, {}).run(run_id)


# --- 2. parallel fan-out and barrier join -----------------------------------


async def test_parallel_nodes_actually_run_concurrently(
    store: JsonlEventStore, run_id: UUID
) -> None:
    """Not just 'both complete' - genuinely overlapping in wall-clock time."""
    graph = WorkflowGraph(
        name="parallel",
        entry=("split",),
        nodes=(
            agent("split"),
            agent("left"),
            agent("right"),
            NodeSpec(id="barrier", kind=NodeKind.BARRIER, join=JoinPolicy.ALL),
            terminal(),
        ),
        edges=(
            Edge(source="split", target="left"),
            Edge(source="split", target="right"),
            Edge(source="left", target="barrier"),
            Edge(source="right", target="barrier"),
            Edge(source="barrier", target="done"),
        ),
    )
    left = SlowExecutor(ok(), delay=0.2)
    right = SlowExecutor(ok(), delay=0.2)
    executors: dict[str, NodeExecutor] = {
        "split": FixedExecutor(ok()),
        "left": left,
        "right": right,
    }

    state = await make_scheduler(graph, store, executors).run(run_id)

    assert state.status is RunStatus.COMPLETED
    # Interval overlap, not a wall-clock threshold: robust under CPU
    # contention or a loaded machine, where a fixed "elapsed < N seconds"
    # assertion would be flaky in either direction.
    assert left.overlaps(right), (
        f"left [{left.started_at}, {left.finished_at}] and right "
        f"[{right.started_at}, {right.finished_at}] did not overlap - not concurrent"
    )


@pytest.mark.parametrize("join", [JoinPolicy.ALL, JoinPolicy.ANY])
async def test_barrier_waits_for_join_policy(
    store: JsonlEventStore, run_id: UUID, join: JoinPolicy
) -> None:
    graph = WorkflowGraph(
        name="join",
        entry=("split",),
        nodes=(
            agent("split"),
            agent("left"),
            agent("right"),
            NodeSpec(id="barrier", kind=NodeKind.BARRIER, join=join),
            terminal(),
        ),
        edges=(
            Edge(source="split", target="left"),
            Edge(source="split", target="right"),
            Edge(source="left", target="barrier"),
            Edge(source="right", target="barrier"),
            Edge(source="barrier", target="done"),
        ),
    )
    executors: dict[str, NodeExecutor] = {
        "split": FixedExecutor(ok()),
        "left": FixedExecutor(ok()),
        "right": FixedExecutor(ok()),
    }
    state = await make_scheduler(graph, store, executors).run(run_id)
    assert state.status is RunStatus.COMPLETED
    assert state.nodes["barrier"].status is NodeStatus.SUCCEEDED


async def test_quorum_join_proceeds_without_waiting_for_a_straggler(
    store: JsonlEventStore, run_id: UUID
) -> None:
    """Quorum's purpose: don't block on every predecessor, just enough of them.

    `c` is deliberately an ungranted approval gate rather than a failure - a
    failure would fail the *run* (a separate, correctly-tested mechanism, see
    `test_failure_with_no_recovery_edge_fails_the_run`), which would stop the
    scheduler before it ever got to prove anything about quorum specifically.
    An indefinitely-pending approval isolates the property actually being
    tested: the barrier proceeds on 2 of 3 without c ever resolving.
    """
    graph = WorkflowGraph(
        name="quorum",
        entry=("split",),
        nodes=(
            agent("split"),
            agent("a"),
            agent("b"),
            NodeSpec(id="c", kind=NodeKind.GATE, requires_approval=True),
            NodeSpec(id="barrier", kind=NodeKind.BARRIER, join=JoinPolicy.QUORUM, quorum=2),
            terminal(),
        ),
        edges=(
            Edge(source="split", target="a"),
            Edge(source="split", target="b"),
            Edge(source="split", target="c"),
            Edge(source="a", target="barrier"),
            Edge(source="b", target="barrier"),
            Edge(source="c", target="barrier"),
            Edge(source="barrier", target="done"),
        ),
    )
    executors: dict[str, NodeExecutor] = {
        "split": FixedExecutor(ok()),
        "a": FixedExecutor(ok()),
        "b": FixedExecutor(ok()),
    }
    # No approval provider: c is a real human who has not decided yet.
    state = await make_scheduler(graph, store, executors, approvals=None).run(run_id)

    assert state.nodes["barrier"].status is NodeStatus.SUCCEEDED
    assert state.status_of("done") is NodeStatus.SUCCEEDED
    assert state.nodes["c"].status is NodeStatus.AWAITING_APPROVAL
    # The run is not COMPLETED while c is still outstanding, even though the
    # barrier already moved on - _is_complete() requires every node settled.
    assert state.status is RunStatus.RUNNING


# --- 3. bounded repair cycle -------------------------------------------------


def _repair_graph(cycle_budget: int) -> WorkflowGraph:
    return WorkflowGraph(
        name="repair",
        entry=("test",),
        nodes=(
            agent("test", cycle_budget=cycle_budget, join=JoinPolicy.ANY),
            agent("repair"),
            terminal(),
        ),
        edges=(
            Edge(source="test", target="repair", condition=EdgeCondition.ON_FAILURE),
            Edge(source="repair", target="test", condition=EdgeCondition.ALWAYS),
            Edge(source="test", target="done"),
        ),
    )


async def test_repair_cycle_succeeds_within_budget(store: JsonlEventStore, run_id: UUID) -> None:
    graph = _repair_graph(cycle_budget=2)
    test_executor = SequenceExecutor([fail("compile error"), ok(kind="TestSuite")])
    executors: dict[str, NodeExecutor] = {
        "test": test_executor,
        "repair": FixedExecutor(ok(kind="CodePatch")),
    }

    state = await make_scheduler(graph, store, executors).run(run_id)

    assert state.status is RunStatus.COMPLETED
    assert test_executor.call_count == 2
    assert state.nodes["test"].attempt == 2
    assert state.nodes["repair"].status is NodeStatus.SUCCEEDED


async def test_repair_cycle_halts_when_budget_exhausted(
    store: JsonlEventStore, run_id: UUID
) -> None:
    graph = _repair_graph(cycle_budget=2)
    always_fails = FixedExecutor(fail("still broken"))
    executors: dict[str, NodeExecutor] = {
        "test": always_fails,
        "repair": FixedExecutor(ok(kind="CodePatch")),
    }

    state = await make_scheduler(graph, store, executors).run(run_id)

    assert state.status is RunStatus.HALTED
    assert state.halt_reason is not None
    assert "cycle_budget" in state.halt_reason
    assert always_fails.call_count == 2  # exactly the budgeted number of attempts
    # 'test' was explicitly re-staged to RETRYING when repair's second success
    # propagated (see Scheduler._propagate_edge_completion) - the halt then
    # happens at the *next* dispatch attempt, one step later, so RETRYING
    # rather than FAILED is what the audit trail correctly shows at halt time.
    assert state.nodes["test"].status is NodeStatus.RETRYING


async def test_a_repair_node_itself_failing_does_not_immediately_fail_the_run(
    store: JsonlEventStore, run_id: UUID
) -> None:
    """`repair`'s only outgoing edge is `condition: always`, never
    `on_failure` (exactly `workflows/greenfield.yaml`'s real
    `repair`/`repair_domain`/`repair_api` shape). Before
    `Scheduler._has_recovery_edge` recognised `ALWAYS` as recovery too, a
    repair agent that itself failed here would have been reported as having
    no recovery edge and immediately emitted `RUN_FAILED` - even though the
    `always` edge had, in the very same call, already rescheduled `test` for
    another attempt. Found while adding `kernel/tools/fs.py`'s shared-file
    conflict guard: `agents/implementer.py`'s bounded reconciliation retry can
    now make a repair position fail outright on a second conflict, and this
    must degrade gracefully, not end the run."""
    graph = _repair_graph(cycle_budget=2)
    test_executor = SequenceExecutor([fail("compile error"), ok(kind="TestSuite")])
    executors: dict[str, NodeExecutor] = {
        "test": test_executor,
        # Fails on its first attempt, succeeds on its second - standing in
        # for `agents/implementer.py`'s reconciliation retry losing once and
        # winning the next time `repair` is dispatched.
        "repair": SequenceExecutor([fail("shared file conflict"), ok(kind="CodePatch")]),
    }

    state = await make_scheduler(graph, store, executors).run(run_id)

    assert state.status is RunStatus.COMPLETED
    assert test_executor.call_count == 2


# --- 4. terminal failure with no recovery edge ------------------------------


async def test_failure_with_no_recovery_edge_fails_the_run(
    store: JsonlEventStore, run_id: UUID
) -> None:
    graph = WorkflowGraph(
        name="no-recovery",
        entry=("solo",),
        nodes=(agent("solo"), terminal()),
        edges=(Edge(source="solo", target="done"),),
    )
    executors: dict[str, NodeExecutor] = {"solo": FixedExecutor(fail("unrecoverable"))}

    state = await make_scheduler(graph, store, executors).run(run_id)

    assert state.status is RunStatus.FAILED
    assert state.nodes["solo"].status is NodeStatus.FAILED
    assert state.nodes["solo"].last_error == "unrecoverable"
    # 'done' was never touched by any event, so it has no NodeState entry at
    # all - status_of() is the correct way to read "still PENDING" for a node
    # the fold never saw, as opposed to dict-indexing state.nodes directly.
    assert state.status_of("done") is NodeStatus.PENDING
    assert "done" not in state.nodes


# --- 5. gate rejection: the one-hop clarification cycle ---------------------


def _clarification_graph(cycle_budget: int) -> WorkflowGraph:
    return WorkflowGraph(
        name="clarify",
        entry=("req",),
        nodes=(
            agent("req", cycle_budget=cycle_budget),
            NodeSpec(id="gate1", kind=NodeKind.GATE, requires_approval=True),
            terminal(),
        ),
        edges=(
            Edge(source="req", target="gate1"),
            Edge(source="gate1", target="req", condition=EdgeCondition.ON_REJECTED),
            Edge(source="gate1", target="done"),
        ),
    )


async def test_rejected_gate_sends_its_producer_back(store: JsonlEventStore, run_id: UUID) -> None:
    graph = _clarification_graph(cycle_budget=3)
    req_executor = SequenceExecutor(
        [ok(kind="RequirementSpec", version=1), ok(kind="RequirementSpec", version=2)]
    )
    approvals = ScriptedApprovals(
        {
            "gate1": [
                ApprovalDecision(granted=False, actor="alice", reason="too vague"),
                ApprovalDecision(granted=True, actor="alice", reason="clear now"),
            ]
        }
    )

    state = await make_scheduler(graph, store, {"req": req_executor}, approvals=approvals).run(
        run_id
    )

    assert state.status is RunStatus.COMPLETED
    assert req_executor.call_count == 2
    assert state.nodes["req"].attempt == 2
    assert state.approvals["gate1"].granted
    # Two distinct artifacts: rejection must not have reused the first hash.
    assert len(state.artifacts) == 2

    events = await store.read_all(run_id)
    stale_events = [
        e for e in events if e.type is EventType.NODE_MARKED_STALE and e.node_id == "req"
    ]
    assert len(stale_events) == 1, (
        "req should be marked stale exactly once, between the reject and the retry"
    )


async def test_clarification_cycle_halts_when_budget_exhausted(
    store: JsonlEventStore, run_id: UUID
) -> None:
    graph = _clarification_graph(cycle_budget=2)
    executors: dict[str, NodeExecutor] = {"req": FixedExecutor(ok(kind="RequirementSpec"))}
    approvals = ScriptedApprovals(
        {
            "gate1": [
                ApprovalDecision(granted=False, actor="alice", reason="no"),
                ApprovalDecision(granted=False, actor="alice", reason="still no"),
            ]
        }
    )

    state = await make_scheduler(graph, store, executors, approvals=approvals).run(run_id)

    assert state.status is RunStatus.HALTED
    assert state.halt_reason is not None
    assert "cycle_budget" in state.halt_reason


async def test_a_rejected_gate_with_no_recovery_edge_fails_cleanly_rather_than_hanging(
    store: JsonlEventStore, run_id: UUID
) -> None:
    """Found via a live run: a gate with no `on_rejected` edge (unlike
    `gate1` above) used to leave `state.status` silently stuck at RUNNING
    forever once rejected - nothing downstream could ever become ready
    (correctly), but nothing ever told the caller the run was actually over.
    A rejected gate with no way forward must report `RunStatus.FAILED`, not
    a `RunStatus.RUNNING` that will never change no matter how many more
    times `run()` is called on this same run_id."""
    graph = WorkflowGraph(
        name="dead_end_rejection",
        entry=("req",),
        nodes=(
            agent("req"),
            NodeSpec(id="gate1", kind=NodeKind.GATE, requires_approval=True),
            agent("arch"),
            terminal(),
        ),
        edges=(
            Edge(source="req", target="gate1"),
            Edge(source="gate1", target="arch"),  # no on_rejected edge at all
            Edge(source="arch", target="done"),
        ),
    )
    executors: dict[str, NodeExecutor] = {
        "req": FixedExecutor(ok(kind="RequirementSpec")),
        "arch": FixedExecutor(ok(kind="DesignSpec")),
    }
    approvals = ScriptedApprovals(
        {"gate1": [ApprovalDecision(granted=False, actor="alice", reason="not correct")]}
    )

    state = await make_scheduler(graph, store, executors, approvals=approvals).run(run_id)

    assert state.status is RunStatus.FAILED
    assert state.nodes["gate1"].status is NodeStatus.REJECTED
    # arch correctly never ran - it has no valid path to readiness either.
    assert "arch" not in state.nodes or state.nodes["arch"].status is NodeStatus.PENDING


async def test_rejection_with_no_decision_leaves_run_awaiting(
    store: JsonlEventStore, run_id: UUID
) -> None:
    """No approvals provider at all == a real human who has not acted yet.

    The run must not hang, spin, or fail - it stops cleanly, still RUNNING,
    ready for a later `run()` call once a decision exists.
    """
    graph = WorkflowGraph(
        name="pending",
        entry=("req",),
        nodes=(
            agent("req"),
            NodeSpec(id="gate1", kind=NodeKind.GATE, requires_approval=True),
            terminal(),
        ),
        edges=(Edge(source="req", target="gate1"), Edge(source="gate1", target="done")),
    )
    executors: dict[str, NodeExecutor] = {"req": FixedExecutor(ok())}

    state = await make_scheduler(graph, store, executors, approvals=None).run(run_id)

    assert state.status is RunStatus.RUNNING
    assert state.nodes["gate1"].status is NodeStatus.AWAITING_APPROVAL
    assert "gate1" in state.approvals
    assert not state.approvals["gate1"].granted


async def test_resuming_after_a_late_approval_completes_the_run(
    store: JsonlEventStore, run_id: UUID
) -> None:
    """The realistic human-in-the-loop shape: run() returns with a node
    awaiting approval, a decision is appended out of band, run() is called
    again and picks up exactly where it left off."""
    graph = WorkflowGraph(
        name="pending-then-decided",
        entry=("req",),
        nodes=(
            agent("req"),
            NodeSpec(id="gate1", kind=NodeKind.GATE, requires_approval=True),
            terminal(),
        ),
        edges=(Edge(source="req", target="gate1"), Edge(source="gate1", target="done")),
    )
    executors: dict[str, NodeExecutor] = {"req": FixedExecutor(ok())}

    first_pass = await make_scheduler(graph, store, executors, approvals=None).run(run_id)
    assert first_pass.status is RunStatus.RUNNING
    assert first_pass.nodes["gate1"].status is NodeStatus.AWAITING_APPROVAL

    approvals = ScriptedApprovals({"gate1": [ApprovalDecision(granted=True, actor="alice")]})
    second_pass = await make_scheduler(graph, store, executors, approvals=approvals).run(run_id)

    assert second_pass.status is RunStatus.COMPLETED
    assert second_pass.approvals["gate1"].granted


# --- 6. budget exhaustion halts the run, not just the node ------------------


async def test_token_budget_exhaustion_halts_before_the_next_node(
    store: JsonlEventStore, run_id: UUID
) -> None:
    graph = WorkflowGraph(
        name="budget",
        entry=("a",),
        nodes=(agent("a"), agent("b"), terminal()),
        edges=(Edge(source="a", target="b"), Edge(source="b", target="done")),
    )
    # `a` reports usage that alone exceeds the budget; `b` must never dispatch.
    executors: dict[str, NodeExecutor] = {
        "a": FixedExecutor(
            NodeExecutionOutcome(ok=True, artifact_kind="X", artifact_payload={"v": 1}, usd=5.0)
        ),
        "b": FixedExecutor(ok(kind="Y")),
    }
    limits = BudgetLimits(max_tokens=10_000_000, max_usd=1.0, max_wallclock_seconds=3600)

    state = await make_scheduler(graph, store, executors, limits=limits).run(run_id)

    assert state.status is RunStatus.HALTED
    assert state.halt_reason is not None
    assert "cost budget" in state.halt_reason
    assert state.nodes["a"].status is NodeStatus.SUCCEEDED
    # 'b' was evaluated at its entry gate and refused there - that is a more
    # informative terminal state than PENDING, and is left as-is (not
    # overwritten) so the audit trail shows exactly where the halt occurred.
    assert state.nodes["b"].status is NodeStatus.ENTRY_GATE


# --- 7. checkpoint / resume equivalence -------------------------------------


async def test_checkpointed_resume_matches_a_full_fold(
    store: JsonlEventStore, run_id: UUID, tmp_path: Path
) -> None:
    """The property checkpoints exist for: resuming from one must reach the
    exact state a from-scratch fold of the same log reaches."""
    graph = _repair_graph(cycle_budget=2)
    executors: dict[str, NodeExecutor] = {
        "test": SequenceExecutor([fail("first try"), ok(kind="TestSuite")]),
        "repair": FixedExecutor(ok(kind="CodePatch")),
    }
    checkpoints = CheckpointStore(tmp_path / "checkpoints")

    resumed_state = await make_scheduler(graph, store, executors, checkpoints=checkpoints).run(
        run_id
    )

    assert resumed_state.status is RunStatus.COMPLETED
    checkpoint = await checkpoints.load(run_id)
    assert checkpoint is not None, "a checkpoint should have been written during the run"

    full_fold = fold(run_id, await store.read_all(run_id))
    via_checkpoint = await resume(store, run_id, checkpoints=checkpoints)

    assert full_fold.model_dump() == via_checkpoint.model_dump()
    assert full_fold.model_dump() == resumed_state.model_dump()


async def test_a_second_scheduler_can_finish_what_the_first_started(
    store: JsonlEventStore, run_id: UUID
) -> None:
    """Simulates a process restart: one Scheduler processes the first node;
    a brand new instance, with no shared in-memory state, finishes the run
    purely by resuming from the event log."""
    graph = WorkflowGraph(
        name="two-step",
        entry=("a",),
        nodes=(agent("a"), agent("b"), terminal()),
        edges=(Edge(source="a", target="b"), Edge(source="b", target="done")),
    )
    executors: dict[str, NodeExecutor] = {
        "a": FixedExecutor(ok(kind="A")),
        "b": FixedExecutor(ok(kind="B")),
    }

    first = make_scheduler(graph, store, executors)
    await first.run_one_step_for_testing(run_id)  # crash-simulation seam, see Scheduler

    mid_state = await resume(store, run_id)
    assert mid_state.nodes["a"].status is NodeStatus.SUCCEEDED
    assert mid_state.status_of("b") is NodeStatus.PENDING

    second = make_scheduler(graph, store, executors)
    final_state = await second.run(run_id)

    assert final_state.status is RunStatus.COMPLETED
    independent_fold = fold(run_id, await store.read_all(run_id))
    assert independent_fold.model_dump() == final_state.model_dump()


# --- declared timeouts are advisory, deliberately -----------------------------


class _SlowerThanDeclared:
    """Takes longer than its node declares. Stands in for the real case: a
    `scaffold` call that legitimately needs 201s against a declared 120s,
    mostly spent in adaptive extended thinking that emits no tokens."""

    def __init__(self) -> None:
        self.finished = False

    async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome:
        await asyncio.sleep(0.05)
        self.finished = True
        return ok()


def _slow_graph() -> WorkflowGraph:
    return WorkflowGraph(
        name="slow",
        entry=("slow",),
        nodes=(
            # Declared far below what the executor above actually takes.
            NodeSpec(id="slow", kind=NodeKind.AGENT, handler="slow", timeout_seconds=0.001),
            NodeSpec(id="done", kind=NodeKind.TERMINAL),
        ),
        edges=(Edge(source="slow", target="done"),),
    )


async def test_a_node_slower_than_its_declared_timeout_is_left_alone(tmp_path: Path) -> None:
    """The scheduler does not cancel on `timeout_seconds`, and this pins that
    as a decision rather than an omission.

    Enforcement was added here briefly and removed after one live run: it
    cancelled `scaffold` at its declared 120s on a call that had succeeded in
    201s the run before, failing the whole run after two human approvals.
    Execution is already bounded at the two layers that can attribute a stall
    to something - `kernel.tools.registry` wraps every tool invocation in
    `asyncio.wait_for(..., spec.timeout_s)`, and the provider SDK bounds each
    HTTP call - and a third ceiling above those could only cut short work
    those layers considered healthy.

    If per-node deadlines are wanted back, they need calibrating against
    `max_tokens` and against `providers/structured.py`'s retry, and every
    agent node needs a recovery edge first. Re-adding `asyncio.wait_for`
    around the executor call will fail this test, which is the point."""
    executor = _SlowerThanDeclared()
    store = JsonlEventStore(tmp_path / "events", fsync=False)

    state = await Scheduler(
        _slow_graph(), store, {"slow": executor}, entry_gate=BudgetEntryGate(GENEROUS)
    ).run(uuid4())

    assert executor.finished is True, "the executor was cancelled mid-flight"
    assert state.nodes["slow"].status is NodeStatus.SUCCEEDED
    assert state.status is RunStatus.COMPLETED
