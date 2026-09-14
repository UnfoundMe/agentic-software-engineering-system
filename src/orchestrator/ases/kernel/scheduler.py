"""The scheduler: readiness evaluation, dispatch, and the run loop.

This is where "the engine owns control flow" becomes an executable property
rather than a design statement. The scheduler decides what runs next by
evaluating the graph against folded state; a `NodeExecutor` is handed a node
and returns a plain outcome - it has no way to influence what happens after.

**What this module does implement:** readiness (join policies over edge
conditions), parallel dispatch of independent agent/tool nodes, entry-gate
budget enforcement, human approval (via a
pluggable `ApprovalProvider`), bounded repair cycles, a *direct* one-hop
mechanism (`_propagate_edge_completion`) that re-stages an already-settled
successor when a matching edge fires - this is what both the repair loop and
a rejected gate sending its producer back run on - dynamic subgraph admission
(`_admit_subgraph_for`, via an injected `SubgraphProvider`; this is what lets
a decomposer's plan become executable nodes mid-run, re-applied from the
event log on resume by `_readmit_recorded_subgraphs`), and safe-stop on
budget or cycle exhaustion.

**What it deliberately does not implement yet**, so nothing here is mistaken
for more than it is:

- **Full re-planning.** `_propagate_edge_completion` restarts exactly the one
  node named by the matching edge. It does not walk the lineage DAG to find
  and invalidate further descendants, and it does not revoke approvals that
  were granted against a now-stale artifact. That is Phase 7
  (`kernel.replan`), needed once a run can amend an artifact *after* other
  nodes have already consumed it - everything this scheduler handles directly
  only ever fires one hop away from the node that just settled.
- **Tool-level recovery** (idempotency-aware retry, external-effect
  compensation, fallback content). Failures here either follow a graph-level
  `ON_FAILURE` edge (if the workflow author provided one) or fail the run.
  Phase 5 (`kernel.recovery`) adds per-tool classification on top of this.
- **Per-node wall-clock timeouts.** `NodeSpec.timeout_seconds` is declared
  on every node of every workflow YAML and is deliberately *not* enforced
  here. Execution is already bounded twice, at the two layers that can
  actually attribute a stall: `kernel.tools.registry` wraps every tool
  invocation in `asyncio.wait_for(..., spec.timeout_s)`, and the Anthropic
  SDK applies a 600s read timeout to each HTTP call. A third ceiling above
  those bounded nothing new - it could only cut short work the lower layers
  considered healthy, which is exactly what it did: it was added briefly,
  and the first live run under it cancelled `scaffold` at 120s on a call
  that had taken 201s (successfully) the run before, failing the whole run
  after two human approvals had already been granted. The declared value is
  kept as the workflow author's documented expectation and as the input
  `agents/planner.py` gives its admitted nodes; what it is not is a
  guarantee the kernel makes. The residual gap, stated rather than papered
  over: an agent node's worst case is `providers/structured.py`'s two calls
  at the SDK ceiling, and `BudgetEntryGate` evaluates wall-clock at node
  *entry*, so neither stops a single pathologically slow node mid-flight.
- **Policy enforcement.** The only gate implemented is the budget check;
  `kernel.policy` (Phase 3) is a separate concern layered on top later.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from ases.kernel.cancellation import CancellationRequestedError, CancelToken
from ases.kernel.checkpoint import Checkpoint, CheckpointStore, resume
from ases.kernel.events import Actor, Event, EventType, UnsealedEvent
from ases.kernel.failures import FailureKind
from ases.kernel.gates import ApprovalDecision, ApprovalProvider, BudgetEntryGate, GateVerdict
from ases.kernel.graph import (
    Edge,
    EdgeCondition,
    JoinPolicy,
    NodeKind,
    NodeSpec,
    WorkflowGraph,
)
from ases.kernel.hashing import digest
from ases.kernel.state import LEGAL_TRANSITIONS, NodeStatus, RunState, RunStatus, apply
from ases.kernel.store.base import EventStore


class ConfigurationError(RuntimeError):
    """The graph names a handler that was never registered.

    Raised rather than silently skipped - a node with no executor is an
    authoring mistake, not a runtime condition the scheduler should paper over.
    """


class NodeExecutionOutcome(BaseModel):
    """What a `NodeExecutor` reports back. Nothing here is a routing decision -
    it is data the scheduler interprets, exactly as an agent's `AgentResult`
    will be once the agent plane exists (Phase 4 reuses this shape)."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    artifact_kind: str | None = None
    artifact_payload: Mapping[str, object] | None = None
    error: str | None = None
    #: Why it failed, classified by the executor that made the call - see
    #: `kernel.failures.FailureKind`. Diagnostic metadata recorded on
    #: `NODE_FAILED`, deliberately **not** an input to routing: the graph's
    #: edges decide what happens next, exactly as before. Ignored entirely
    #: when `ok` is True.
    failure_kind: FailureKind = FailureKind.ORCHESTRATION_FAILURE
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0


class SubgraphProposal(BaseModel):
    """Nodes and edges a node's output asks to add to the running graph.

    Nothing here is a routing *decision*: a proposal names work to admit, and
    the scheduler still decides what becomes ready and when, from the same
    edge/join evaluation it applies to statically declared nodes. The
    proposing component cannot dispatch anything, cannot reorder anything,
    and cannot admit a node that `WorkflowGraph.validate_graph` rejects.
    """

    model_config = ConfigDict(frozen=True)

    nodes: tuple[NodeSpec, ...]
    edges: tuple[Edge, ...]


@runtime_checkable
class SubgraphProvider(Protocol):
    """Turns a settled node's artifact into a subgraph proposal, or `None`
    when that node admits nothing (which is every node but one).

    This exists so the kernel can admit a decomposer's plan without importing
    the agent plane or knowing what a `TaskGraph` is: the concrete
    implementation (`agents.planner.TaskGraphSubgraphProvider`) lives in the
    agent plane where the contracts do, and is injected. `kernel/` stays
    free of `contracts`/`agents` imports, which
    `tests/invariants/test_layering.py` enforces.
    """

    def propose(self, node: NodeSpec, state: RunState) -> SubgraphProposal | None: ...


@runtime_checkable
class NodeExecutor(Protocol):
    """Performs one node's work. Stateless with respect to the graph: it is
    handed the node and the current state and returns an outcome - it never
    sees the event store and cannot emit events itself."""

    async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome: ...


#: Statuses the generic readiness scan may pick up directly. Deliberately
#: **excludes** SUCCEEDED, FAILED and REJECTED, even though the state machine
#: permits SUCCEEDED->STALE, FAILED->RETRYING and REJECTED->RETRYING -
#: because all three of those predecessor/self statuses are "sticky": once
#: reached, they never change again on their own, so a join check against a
#: permanently-settled predecessor would stay satisfied forever. A generic
#: scan that treated them as re-eligible would restart the node on every
#: subsequent pass for as long as the run stays open for any other reason -
#: not a rare edge case, but the guaranteed outcome for a repair loop or a
#: clarification cycle followed by anything that keeps the run from
#: completing in the same pass. That was exactly the shape of the first bug
#: found while testing this scheduler: `test_linear_run_with_granted_approval_completes`
#: hung outright, and several other tests silently reached the right answer
#: by accident because the run happened to complete before a second, spurious
#: pass could run.
#:
#: REJECTED joined this exclusion later than the other two, found the same
#: way: a gate rejected with no `on_rejected` recovery edge (or whose
#: producer had not yet re-settled) stayed "sticky-eligible" against its own
#: permanently-succeeded predecessor, so the next pass re-dispatched the
#: *gate itself* - re-requesting an approval nobody asked for again, and if
#: no further decision was scripted/available, silently overwriting the
#: rejection's own `ApprovalRecord` (actor, reason) with the blank
#: placeholder `APPROVAL_REQUESTED` sets, corrupting the audit trail of what
#: a human actually decided. `test_greenfield_partial_e2e.py`'s rejection
#: test caught it - a scenario the original clarification-cycle test never
#: exercised, since it always scripted a second, eventually-approving
#: decision and never inspected the intermediate approval record's contents.
#:
#: The fix: SUCCEEDED, FAILED and REJECTED nodes are moved to
#: STALE/RETRYING/RETRYING (respectively) **only** by the explicit, one-shot
#: `_propagate_edge_completion` below, called right when a predecessor
#: settles - never inferred by scanning current status.
_DIRECTLY_ELIGIBLE_STATUSES = frozenset(
    {
        NodeStatus.PENDING,
        NodeStatus.RETRYING,
        NodeStatus.STALE,
        NodeStatus.BLOCKED,
    }
)


def _is_directly_eligible(status: NodeStatus) -> bool:
    return status in _DIRECTLY_ELIGIBLE_STATUSES and NodeStatus.READY in LEGAL_TRANSITIONS[status]


#: A predecessor in one of these statuses has reached a genuine, final
#: outcome for its current attempt - not merely "not running right now"
#: (`AWAITING_APPROVAL`, `STALE` and `RETRYING` are deliberately excluded:
#: each is specifically designed to be reconsidered). Used by
#: `Scheduler._is_permanently_unreachable` below. Deliberately **not** a
#: transitive-closure computation over `LEGAL_TRANSITIONS`: the abstract
#: state machine permits long chains of legal transitions that will never
#: actually happen without a real, separately-triggered event (an early,
#: broader version of this check treated the mere existence of *some* legal
#: path to a matching status as "might still fire", which made nearly every
#: node "possibly reachable" forever - the state machine describes what
#: transitions are *permitted*, not what will spontaneously occur). The
#: right question is narrower: once a node settles here,
#: `_propagate_edge_completion` has already had its one synchronous shot at
#: firing every matching outgoing edge; if a given edge didn't match then, it
#: will not later, absent a new, separately-triggered event this check would
#: see reflected in a *different* status by the time it runs again.
_FINAL_OUTCOME_STATUSES = frozenset(
    {
        NodeStatus.SUCCEEDED,
        NodeStatus.FAILED,
        NodeStatus.REJECTED,
        NodeStatus.CANCELLED,
        NodeStatus.SKIPPED,
        NodeStatus.HALTED,
        NodeStatus.ROLLED_BACK,
        NodeStatus.FALLBACK,
    }
)


def _handler_of(node: NodeSpec) -> str:
    """NodeSpec's validator guarantees a handler for AGENT/TOOL kinds; this
    only narrows the type for the executor lookup."""
    assert node.handler is not None
    return node.handler


class Scheduler:
    """Drives one run to completion, or until it needs a human, halts, or fails.

    One `Scheduler` instance handles one run at a time: `run()` loads state at
    the start of the call and mutates it in place as events are appended, on
    the assumption that this instance is the run's sole writer for the
    duration of the call. Concurrent runs need one `Scheduler` each - cheap
    to construct, since all the expensive state lives in the store.
    """

    def __init__(
        self,
        graph: WorkflowGraph,
        store: EventStore,
        executors: Mapping[str, NodeExecutor],
        *,
        entry_gate: BudgetEntryGate,
        approvals: ApprovalProvider | None = None,
        checkpoints: CheckpointStore | None = None,
        cancel_token: CancelToken | None = None,
        subgraphs: SubgraphProvider | None = None,
    ) -> None:
        #: Mutable **only** through `_admit_subgraph_for`, which replaces it
        #: with a `with_subgraph` result that has already passed
        #: `validate_graph`. Every other read of `self.graph` - readiness,
        #: propagation, completion - sees a graph that is valid by
        #: construction, exactly as it did when this was assigned once.
        self.graph = graph
        self.store = store
        self.executors = executors
        self.entry_gate = entry_gate
        self.approvals = approvals
        self.checkpoints = checkpoints
        self.cancel_token = cancel_token or CancelToken()
        self.subgraphs = subgraphs
        self._run_id: UUID | None = None
        self._state: RunState | None = None

    # -- public entry point ---------------------------------------------

    async def run(self, run_id: UUID) -> RunState:
        state = await resume(self.store, run_id, checkpoints=self.checkpoints)
        self._run_id = run_id
        self._state = state
        self._readmit_recorded_subgraphs()

        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.HALTED):
            return state  # resuming a finished run is a no-op, not an error

        try:
            await self._ensure_started()

            while True:
                self.cancel_token.raise_if_cancelled()
                progressed = await self._step()
                # Checked in this order deliberately: a halt or failure the
                # step just emitted must never be papered over by also
                # reporting completion, even if every node happens to have
                # settled by coincidence in the same pass.
                if state.status is not RunStatus.RUNNING:
                    break  # a RUN_FAILED / RUN_HALTED was already emitted
                if self._is_complete():
                    await self._emit(EventType.RUN_COMPLETED)
                    break
                if not progressed:
                    if not self._awaiting_human_decision():
                        # Genuinely stuck, not just paused: nothing is ready,
                        # nothing is waiting on a human who might still answer,
                        # and the run is not complete - a rejected gate with no
                        # `on_rejected` edge is the common cause (found via a
                        # live run: rejecting a gate with no such edge left
                        # `state.status` silently stuck at RUNNING forever,
                        # since neither RUN_COMPLETED nor any failure event
                        # was ever emitted for it). A later `run()` call on
                        # this same run_id cannot change this outcome either,
                        # unlike the awaiting-approval case below.
                        await self._emit(
                            EventType.RUN_FAILED,
                            reason=(
                                "the run is stuck: no node is ready, none are "
                                "awaiting a human decision, and the run has not "
                                "completed - likely a rejected gate or barrier "
                                "with no recovery edge from here"
                            ),
                        )
                    break  # quiescent: awaiting a human, or nothing left ready
                if self.checkpoints is not None:
                    await self.checkpoints.save(
                        Checkpoint(run_id=run_id, seq=state.last_seq, state=state)
                    )
        except CancellationRequestedError as exc:
            await self._emit(EventType.RUN_HALTED, reason=str(exc))

        return state

    async def run_one_step_for_testing(self, run_id: UUID) -> RunState:
        """Bootstrap the run and process exactly one scheduling step, then
        return - standing in for a process crash immediately afterwards.

        Exists so resumability tests can simulate "the process died mid-run"
        without reaching into private attributes. Not part of the scheduler's
        real operating contract: production code always calls `run()`.
        """
        state = await resume(self.store, run_id, checkpoints=self.checkpoints)
        self._run_id = run_id
        self._state = state
        self._readmit_recorded_subgraphs()
        await self._ensure_started()
        await self._step()
        return state

    async def _ensure_started(self) -> None:
        """Emit `RUN_CREATED` (carrying the workflow name) exactly once, for
        a run that has no events at all yet, then `RUN_STARTED`.

        Split from a single `RUN_STARTED` because `kernel.state`'s fold reads
        `workflow` only from `RUN_CREATED` - a run driven purely by repeated
        `Scheduler.run()` calls with no separate "create the run" step ahead
        of it would otherwise never have its workflow name recorded at all.
        A run whose creation was already recorded by an external caller
        (`last_seq > 0`, status still CREATED) only gets `RUN_STARTED` here,
        not a second `RUN_CREATED`.
        """
        assert self._state is not None
        if self._state.status is not RunStatus.CREATED:
            return
        if self._state.last_seq == 0:
            await self._emit(EventType.RUN_CREATED, workflow=self.graph.name)
        await self._emit(EventType.RUN_STARTED)

    # -- readiness --------------------------------------------------------

    def _compute_ready(self) -> tuple[str, ...]:
        """Nodes eligible to advance toward RUNNING on this pass.

        Two independent rules, checked in this order:

        1. **A declared entry point starts unconditionally the first time it
           is PENDING** - regardless of whether it *also* has incoming edges.
           `req` in a clarification-cycle graph is both: it is where the run
           begins, and it is the target of `gate1`'s `ON_REJECTED` edge for
           every subsequent round. Checking the join first would leave it
           permanently ineligible, since nothing has rejected anything yet on
           the very first pass - there is no predecessor status to match.
           This is genuinely a distinct bug from the "zero incoming edges"
           case below, caught by the very simplest clarification-cycle test
           emitting no events at all past `RUN_STARTED`.
        2. **Otherwise, a node with no incoming edges is never reconsidered**
           once it leaves PENDING - nothing can legitimately drive a restart,
           because there is no edge whose condition could fire to trigger it.
           A node *with* incoming edges is reconsidered only while it sits in
           one of `_DIRECTLY_ELIGIBLE_STATUSES` and its join is satisfied;
           SUCCEEDED and FAILED are deliberately excluded there because both
           are "sticky" and would otherwise re-satisfy a join forever - see
           that set's docstring for the hang this caused before the fix.
        """
        assert self._state is not None
        ready: list[str] = []
        for node in self.graph.nodes:
            current = self._state.status_of(node.id)
            if current is NodeStatus.PENDING and node.id in self.graph.entry:
                ready.append(node.id)
                continue
            incoming = self.graph.predecessors(node.id)
            if not incoming:
                continue
            if not _is_directly_eligible(current):
                continue
            if self._join_satisfied(incoming, node.join, node.quorum):
                ready.append(node.id)
        return tuple(sorted(ready))

    def _join_satisfied(
        self, incoming: tuple[Edge, ...], join: JoinPolicy, quorum: int | None
    ) -> bool:
        assert self._state is not None
        active = [e for e in incoming if e.condition.matches(self._state.status_of(e.source))]
        match join:
            case JoinPolicy.ALL:
                return len(active) == len(incoming)
            case JoinPolicy.ANY:
                return len(active) >= 1
            case JoinPolicy.QUORUM:
                assert quorum is not None
                return len(active) >= quorum
        raise AssertionError(f"unhandled join policy {join}")  # pragma: no cover

    def _has_recovery_edge(self, node_id: str) -> bool:
        """True if some successor edge will act on `node_id` failing - either
        an explicit `ON_FAILURE` edge, or an `ALWAYS` edge, which (per
        `EdgeCondition.matches`) fires on failure too, since "always" is the
        union of every other condition's match set.

        Found by reasoning about `workflows/greenfield.yaml`'s own
        `repair`/`repair_domain`/`repair_api` nodes: each has exactly one
        outgoing edge, `condition: always`, back to the node it is repairing -
        never `on_failure`. Before this fix, a repair agent that itself
        failed (an LLM/schema error, or a tool call refused for a reason
        unrelated to what it was repairing) would have been reported as
        having "no recovery edge" and immediately failed the whole run with
        `RUN_FAILED` - even though `_propagate_edge_completion` had, moments
        earlier in the very same call, already fired that `always` edge and
        rescheduled its target for another attempt. Never triggered by a live
        run (no repair agent has failed outright yet), but is exactly what
        `kernel/tools/fs.py`'s shared-file conflict guard can now cause on
        purpose - see `agents/implementer.py`'s bounded reconciliation retry -
        so it had to be correct before that guard could rely on failing
        gracefully instead of ending the run."""
        return any(
            e.condition in (EdgeCondition.ON_FAILURE, EdgeCondition.ALWAYS)
            for e in self.graph.successors(node_id)
        )

    def _latest_artifact(self, node_id: str) -> str:
        state_for_node = self._state.nodes.get(node_id) if self._state else None
        return state_for_node.produced[-1] if state_for_node and state_for_node.produced else ""

    def _input_artifacts(self, node: NodeSpec) -> tuple[str, ...]:
        return tuple(
            h
            for edge in self.graph.predecessors(node.id)
            if (h := self._latest_artifact(edge.source))
        )

    # -- one scheduling step ----------------------------------------------

    async def _step(self) -> bool:
        assert self._state is not None
        # A node left AWAITING_APPROVAL by an earlier `run()` call is not
        # something `_compute_ready` can ever pick up - it is not eligible to
        # move toward READY at all, it is waiting on a decision that may now
        # exist. Checked first, every step, since a decision can arrive
        # between any two calls to `run()`.
        progressed = await self._recheck_pending_approvals()

        ready_ids = self._compute_ready()
        if not ready_ids:
            return progressed

        dispatch: list[NodeSpec] = []
        for node_id in ready_ids:
            node = self.graph.node(node_id)
            entered = await self._stage_and_enter(node)
            if not entered:
                return False  # budget or cycle exhausted; run was halted

            if node.kind in (NodeKind.AGENT, NodeKind.TOOL):
                # NodeSpec's validator guarantees a handler for these kinds;
                # this assert only narrows the type for the lookup below.
                assert node.handler is not None
                if node.handler not in self.executors:
                    raise ConfigurationError(
                        f"node {node.id!r} declares handler {node.handler!r}, "
                        "which has no registered NodeExecutor"
                    )
                dispatch.append(node)
            elif node.kind is NodeKind.GATE and node.requires_approval:
                artifact_hash = self._latest_artifact_of_predecessor(node)
                await self._request_approval(node, artifact_hash)
            else:
                await self._auto_complete(node)

        if dispatch:
            # Genuinely concurrent: the executor calls (standing in for LLM or
            # tool latency) run together via gather. Only the *outcomes* are
            # applied sequentially afterwards, since event emission mutates
            # shared state and must not interleave.
            assert self._state is not None
            outcomes = await asyncio.gather(
                *(self.executors[_handler_of(n)].execute(n, self._state) for n in dispatch)
            )
            for node, outcome in zip(dispatch, outcomes, strict=True):
                await self._finish_agent_node(node, outcome)

        return True

    async def _stage_and_enter(self, node: NodeSpec) -> bool:
        """Advance `node` from its current (already-eligible) status through
        to RUNNING.

        Any staging transition out of SUCCEEDED or FAILED has already
        happened, once, via `_propagate_edge_completion` when the relevant
        predecessor settled - by the time a node reaches this method it is
        already in one of `_DIRECTLY_ELIGIBLE_STATUSES`.

        Returns False (and halts the run) if the node's cycle budget or the
        run's resource budget is exhausted - in both cases the run stops
        rather than proceeding on a node that has exceeded what the workflow
        author declared as safe.
        """
        assert self._state is not None
        next_attempt = self._state.node(node.id).attempt + 1
        if node.cycle_budget is not None and next_attempt > node.cycle_budget:
            await self._emit(
                EventType.RUN_HALTED,
                reason=(
                    f"node {node.id!r} exhausted its cycle_budget "
                    f"({node.cycle_budget}); safe-stopping rather than continuing"
                ),
            )
            return False

        await self._emit(EventType.NODE_READY, node_id=node.id)

        gate_result = self.entry_gate.evaluate(self._state)
        await self._emit(
            EventType.NODE_ENTRY_GATE,
            node_id=node.id,
            verdict=gate_result.verdict,
            reason=gate_result.reason,
        )
        if gate_result.verdict != GateVerdict.PASS:
            await self._emit(EventType.RUN_HALTED, reason=gate_result.reason)
            return False

        await self._emit(EventType.NODE_STARTED, node_id=node.id, attempt=next_attempt)
        return True

    async def _auto_complete(self, node: NodeSpec) -> None:
        """BARRIER, TERMINAL, and non-approval GATE nodes: no executor, no
        artifact - reaching them successfully is the entire outcome."""
        attempt = self._state.node(node.id).attempt if self._state else 0
        await self._emit(
            EventType.NODE_EXIT_GATE, node_id=node.id, attempt=attempt, verdict=GateVerdict.PASS
        )
        await self._emit(EventType.NODE_SUCCEEDED, node_id=node.id, attempt=attempt)
        await self._propagate_edge_completion(node.id)

    async def _request_approval(self, node: NodeSpec, artifact_hash: str) -> None:
        """Request a human decision on `artifact_hash` and act on it if one is
        available immediately.

        Used both by approval-gate nodes (approving a *predecessor's* output)
        and by agent/tool nodes declared `requires_approval` (approving their
        *own* just-produced output) - the two differ only in which hash the
        caller passes in.
        """
        assert self._state is not None
        await self._emit(EventType.APPROVAL_REQUESTED, node_id=node.id, artifact_hash=artifact_hash)

        decision = await self.approvals.decide(node, self._state) if self.approvals else None
        if decision is None:
            return  # left AWAITING_APPROVAL; a later `run()` call may resolve it
        await self._apply_approval_decision(node.id, artifact_hash, decision)

    async def _recheck_pending_approvals(self) -> bool:
        """Ask the provider again for every node still AWAITING_APPROVAL.

        This is the counterpart to the "left AWAITING_APPROVAL" branch above:
        without it, a decision that arrives between two separate `run()`
        calls - the realistic human-in-the-loop shape - would never be
        noticed, because `_compute_ready` deliberately never looks at
        AWAITING_APPROVAL (it is not a status a node can be *dispatched*
        from; it is a status a node is *waiting* in).
        """
        assert self._state is not None
        if self.approvals is None:
            return False
        progressed = False
        awaiting = sorted(
            node_id
            for node_id, node_state in self._state.nodes.items()
            if node_state.status is NodeStatus.AWAITING_APPROVAL
        )
        for node_id in awaiting:
            node = self.graph.node(node_id)
            record = self._state.approvals.get(node_id)
            artifact_hash = record.artifact_hash if record else ""
            decision = await self.approvals.decide(node, self._state)
            if decision is None:
                continue
            await self._apply_approval_decision(node_id, artifact_hash, decision)
            progressed = True
        return progressed

    async def _apply_approval_decision(
        self, node_id: str, artifact_hash: str, decision: ApprovalDecision
    ) -> None:
        event_type = EventType.APPROVAL_GRANTED if decision.granted else EventType.APPROVAL_REJECTED
        await self._emit(
            event_type,
            node_id=node_id,
            artifact_hash=artifact_hash,
            reason=decision.reason,
            actor=Actor.human(decision.actor),
        )
        await self._propagate_edge_completion(node_id)

    def _latest_artifact_of_predecessor(self, node: NodeSpec) -> str:
        for edge in self.graph.predecessors(node.id):
            if artifact := self._latest_artifact(edge.source):
                return artifact
        return ""

    async def _propagate_edge_completion(self, node_id: str) -> None:
        """After `node_id` settles (succeeded / failed / rejected), explicitly
        re-stage any successor that has *already run* and would otherwise
        never be reconsidered by the generic readiness scan - which
        deliberately excludes SUCCEEDED and FAILED (see
        `_DIRECTLY_ELIGIBLE_STATUSES`) precisely so that a permanently
        satisfied edge cannot re-trigger its target forever.

        This single mechanism is what both the bounded repair cycle
        (FAILED -[on_failure]-> repair -[always]-> the failed node, now
        RETRYING) and the one-hop clarification cycle
        (REJECTED -[on_rejected]-> its producer, now STALE) run on. It fires
        exactly once per settling event, because it is called exactly once,
        right when that event is emitted - never inferred from a scan.

        Deliberately narrow, as the module docstring says: this restarts only
        the specific node named by a matching edge. It does not walk the
        lineage DAG to find further descendants - that is Phase 7.
        """
        assert self._state is not None
        current_status = self._state.status_of(node_id)
        for edge in self.graph.successors(node_id):
            if not edge.condition.matches(current_status):
                continue
            target_status = self._state.status_of(edge.target)
            if target_status is NodeStatus.SUCCEEDED:
                await self._emit(EventType.NODE_MARKED_STALE, node_id=edge.target)
            elif target_status in (NodeStatus.FAILED, NodeStatus.REJECTED):
                # A REJECTED gate's own producer just settled again (this is
                # what makes the clarification cycle's second pass fire) -
                # re-stage the gate itself, one-shot, exactly like a FAILED
                # node's retry. See `_DIRECTLY_ELIGIBLE_STATUSES`'s docstring
                # for the corrupted-approval-record bug this replaces.
                await self._emit(EventType.NODE_RETRY_SCHEDULED, node_id=edge.target)
            # PENDING targets are already picked up by the normal scan; a
            # target that is currently in flight cannot be safely restaged
            # out from under itself, so it is left alone.

    async def _finish_agent_node(self, node: NodeSpec, outcome: NodeExecutionOutcome) -> None:
        assert self._state is not None
        attempt = self._state.node(node.id).attempt

        if outcome.input_tokens or outcome.output_tokens or outcome.usd:
            await self._emit(
                EventType.LLM_COMPLETED,
                node_id=node.id,
                attempt=attempt,
                input_tokens=outcome.input_tokens,
                output_tokens=outcome.output_tokens,
                usd=outcome.usd,
            )

        if not outcome.ok:
            await self._emit(
                EventType.NODE_FAILED,
                node_id=node.id,
                attempt=attempt,
                error=outcome.error or "unknown error",
                failure_kind=outcome.failure_kind.value,
            )
            await self._propagate_edge_completion(node.id)
            if not self._has_recovery_edge(node.id):
                await self._emit(
                    EventType.RUN_FAILED,
                    reason=(
                        f"node {node.id!r} failed ({outcome.failure_kind.value}) with no "
                        f"ON_FAILURE recovery edge: {outcome.error}"
                    ),
                )
            return

        artifact_hash = ""
        if outcome.artifact_kind is not None:
            # Content-addressed: the hash is a pure function of the payload,
            # not of which node or attempt produced it, so identical output
            # from two attempts is recognised as the same artifact.
            artifact_hash = digest(dict(outcome.artifact_payload or {}))
            await self._emit(
                EventType.ARTIFACT_PRODUCED,
                node_id=node.id,
                attempt=attempt,
                artifact_hash=artifact_hash,
                kind=outcome.artifact_kind,
                inputs=list(self._input_artifacts(node)),
                # Full content, not just the hash - this is what lets a
                # downstream node (or a human at an approval gate) actually
                # read what a producing node made, via
                # `context.retriever.ContextRetriever.fetch`. The event log
                # remains the only place this is stored, consistent with
                # "everything is derived from the event log" - no separate
                # artifact store, no second place content could drift from
                # what was actually produced.
                content=dict(outcome.artifact_payload or {}),
            )
            await self._emit(
                EventType.ARTIFACT_VALIDATED, node_id=node.id, artifact_hash=artifact_hash
            )

        # Admission happens here - after the artifact exists in folded state
        # (the provider reads it from there) and *before* the exit gate, so a
        # proposal the graph refuses fails this node like any other bad
        # output instead of having to un-succeed a node that already passed.
        if not await self._admit_subgraph_for(node, attempt):
            return

        await self._emit(
            EventType.NODE_EXIT_GATE, node_id=node.id, attempt=attempt, verdict=GateVerdict.PASS
        )

        if node.requires_approval:
            await self._request_approval(node, artifact_hash)
        else:
            await self._emit(EventType.NODE_SUCCEEDED, node_id=node.id, attempt=attempt)
            await self._propagate_edge_completion(node.id)

    # -- dynamic subgraph admission -----------------------------------------

    async def _admit_subgraph_for(self, node: NodeSpec, attempt: int) -> bool:
        """Ask the injected provider whether `node`'s output admits new nodes;
        admit them if the graph accepts them. Returns False if the node must
        be failed instead.

        This is the mechanism `kernel/scheduler.py`'s own docstring used to
        list under "deliberately does not implement yet". The bound on it is
        `WorkflowGraph.with_subgraph`, unchanged: a proposal is validated as a
        whole candidate graph - no dangling edge, no duplicate id, no
        unreachable node, no node that cannot reach a terminal, no unbounded
        cycle - before it can execute. A rejected proposal never runs, and
        the rejection is recorded (`SUBGRAPH_REJECTED`) rather than silently
        dropped.

        The admitted nodes and edges are written into the `SUBGRAPH_ADMITTED`
        payload in full, not just their ids. That is what makes a run
        resumable across processes: `_readmit_recorded_subgraphs` rebuilds the
        same graph from the event log on the next `run()`, so a run that
        crashed after admission does not come back with the static YAML's
        graph and a state full of nodes that graph has never heard of.
        """
        if self.subgraphs is None:
            return True
        assert self._state is not None
        try:
            proposal = self.subgraphs.propose(node, self._state)
        except ValueError as exc:
            # The provider refused to build a proposal at all - see the
            # `except ValueError` below for why this is caught by base type.
            await self._emit(EventType.SUBGRAPH_REJECTED, node_id=node.id, reason=str(exc))
            await self._emit(
                EventType.NODE_FAILED,
                node_id=node.id,
                attempt=attempt,
                error=f"proposed plan could not be admitted: {exc}",
                failure_kind=FailureKind.ORCHESTRATION_FAILURE,
            )
            await self._propagate_edge_completion(node.id)
            if not self._has_recovery_edge(node.id):
                await self._emit(
                    EventType.RUN_FAILED,
                    reason=f"node {node.id!r} proposed a plan that cannot be executed: {exc}",
                )
            return False
        if proposal is None:
            return True

        await self._emit(
            EventType.SUBGRAPH_PROPOSED,
            node_id=node.id,
            node_ids=[n.id for n in proposal.nodes],
            edge_count=len(proposal.edges),
        )
        try:
            candidate = self.graph.with_subgraph(proposal.nodes, proposal.edges)
        except ValueError as exc:
            # `GraphError` (the graph refused the shape) and any
            # `ValueError` the provider itself raises while building the
            # proposal - `agents.planner.SubgraphPlanError`, for a plan that
            # names a project the scaffold never created or leaves one
            # nobody implements. Caught as `ValueError` rather than by
            # concrete type because `kernel/` must not import the agent plane
            # (tests/invariants/test_layering.py); both are deliberately
            # `ValueError` subclasses for exactly this reason.
            await self._emit(EventType.SUBGRAPH_REJECTED, node_id=node.id, reason=str(exc))
            await self._emit(
                EventType.NODE_FAILED,
                node_id=node.id,
                attempt=attempt,
                error=f"proposed subgraph was rejected: {exc}",
                failure_kind=FailureKind.ORCHESTRATION_FAILURE,
            )
            await self._propagate_edge_completion(node.id)
            if not self._has_recovery_edge(node.id):
                await self._emit(
                    EventType.RUN_FAILED,
                    reason=f"node {node.id!r} proposed a subgraph the graph rejected: {exc}",
                )
            return False

        self.graph = candidate
        await self._emit(
            EventType.SUBGRAPH_ADMITTED,
            node_id=node.id,
            node_ids=[n.id for n in proposal.nodes],
            nodes=[n.model_dump(mode="json") for n in proposal.nodes],
            edges=[e.model_dump(mode="json") for e in proposal.edges],
        )
        return True

    def _readmit_recorded_subgraphs(self) -> None:
        """Re-apply every subgraph this run already admitted, from the folded
        event log, so a resumed run executes the graph it was actually
        running rather than the static file it started from.

        Re-validated on the way in exactly like a fresh proposal: an export
        that was tampered with cannot smuggle in a node shape `validate_graph`
        would refuse. Idempotent - a subgraph whose nodes are already present
        is skipped, so calling this on an in-process resume is harmless.
        """
        assert self._state is not None
        for record in self._state.admitted_subgraphs:
            nodes = tuple(NodeSpec.model_validate(n) for n in record.get("nodes", ()))
            edges = tuple(Edge.model_validate(e) for e in record.get("edges", ()))
            if not nodes or all(n.id in self.graph.by_id for n in nodes):
                continue
            self.graph = self.graph.with_subgraph(nodes, edges)

    # -- completion ---------------------------------------------------------

    def _awaiting_human_decision(self) -> bool:
        """True while at least one node is `AWAITING_APPROVAL` - a later
        `run()` call on this same run_id could still resolve it once a human
        decides, so a quiescent step here must not be reported as a failure.
        This is the one case `run()`'s stuck-detection must not fire on."""
        assert self._state is not None
        return any(
            node.status is NodeStatus.AWAITING_APPROVAL for node in self._state.nodes.values()
        )

    def _is_complete(self) -> bool:
        assert self._state is not None
        settled = (NodeStatus.SUCCEEDED, NodeStatus.SKIPPED, NodeStatus.CANCELLED)
        return all(
            self._state.status_of(n.id) in settled or self._is_permanently_unreachable(n.id)
            for n in self.graph.nodes
        )

    #: The two statuses `_is_permanently_unreachable` will actually evaluate -
    #: see its own docstring for why `FAILED` joined `PENDING` here.
    _UNREACHABILITY_CANDIDATE_STATUSES = frozenset({NodeStatus.PENDING, NodeStatus.FAILED})

    def _is_permanently_unreachable(self, node_id: str) -> bool:
        """A `PENDING` or `FAILED` node whose every incoming edge's source has
        already settled into a final outcome (`_FINAL_OUTCOME_STATUSES`)
        without that edge's condition matching - e.g. `repair`'s only edge is
        `test_run -[on_failure]-> repair`, and this run's `test_run` already
        `SUCCEEDED`. Such a node was a real, valid branch this run's actual
        path simply never took (if `PENDING`) or took but did not settle
        successfully on its one dispatched attempt (if `FAILED`) - either way
        nothing can ever dispatch it again, and it must not block completion:
        `_is_complete()` would otherwise wait forever for it.

        Found via `test_greenfield_full_e2e.py`: no prior test had a graph
        both large enough to contain a genuinely optional branch and
        expected to reach `RUN_COMPLETED` rather than `HALTED`/`FAILED`, so
        this was never exercised.

        **`FAILED` joined `PENDING` here** while fixing `_has_recovery_edge`
        to recognise an `ALWAYS` edge as recovery (see that method's own
        docstring): `workflows/greenfield.yaml`'s `repair`/`repair_domain`/
        `repair_api` each have exactly one outgoing edge, `condition:
        always`, back to the node they repair - never a second edge back to
        themselves. A repair agent that fails once (which
        `agents/implementer.py`'s bounded reconciliation retry can now
        legitimately do) settles at `FAILED` *permanently* - `LEGAL_TRANSITIONS`
        only ever moves a `FAILED` node again via `_propagate_edge_completion`'s
        one-shot restaging, and nothing targets `repair*` itself for a retry.
        Before this, if the node it repaired then succeeded on its own
        rescheduled attempt, `_is_complete()` would never see the *whole run*
        as complete - `FAILED` is not in `settled`, and the old
        `PENDING`-only guard here refused to even consider it - so the
        scheduler would spin until the "nothing ready, nothing awaiting a
        human" stuck-detection path in `run()` eventually reported
        `RUN_FAILED` anyway, defeating the whole point of letting the repair
        fail gracefully in the first place. Safe to evaluate here regardless
        of *why* a node is `FAILED`: `_is_complete()` (and therefore this
        method) is only ever reached while `state.status is RUNNING`, which
        means every `FAILED` node still present necessarily had a recovery
        edge when it failed - a `FAILED` node with none already ended the run
        via `RUN_FAILED` before `_is_complete()` could ever run.
        """
        assert self._state is not None
        if self._state.status_of(node_id) not in self._UNREACHABILITY_CANDIDATE_STATUSES:
            return False
        incoming = self.graph.predecessors(node_id)
        if not incoming:
            return False  # an entry node with no predecessors is reachable by definition
        for edge in incoming:
            source_status = self._state.status_of(edge.source)
            if source_status not in _FINAL_OUTCOME_STATUSES:
                return False  # predecessor hasn't settled yet - could still go either way
            if edge.condition.matches(source_status):
                return False  # this edge is satisfied; the node should already be ready
        return True

    # -- event emission -------------------------------------------------

    async def _emit(
        self,
        event_type: EventType,
        *,
        node_id: str | None = None,
        attempt: int | None = None,
        actor: Actor | None = None,
        **payload: object,
    ) -> Event:
        assert self._run_id is not None
        assert self._state is not None
        unsealed = UnsealedEvent(
            run_id=self._run_id,
            type=event_type,
            actor=actor or Actor.kernel(),
            node_id=node_id,
            attempt=attempt,
            payload=payload,
        )
        sealed = await self.store.append(unsealed)
        apply(self._state, sealed)
        return sealed
