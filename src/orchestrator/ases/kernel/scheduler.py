"""The scheduler: readiness evaluation, dispatch, and the run loop.

This is where "the engine owns control flow" becomes an executable property
rather than a design statement. The scheduler decides what runs next by
evaluating the graph against folded state; a `NodeExecutor` is handed a node
and returns a plain outcome - it has no way to influence what happens after.

**What this module does implement:** readiness (join policies over edge
conditions), parallel dispatch of independent agent/tool nodes, entry-gate
budget enforcement, human approval (via a pluggable `ApprovalProvider`),
bounded repair cycles, a *direct* one-hop mechanism (`_propagate_edge_completion`)
that re-stages an already-settled successor when a matching edge fires - this
is what both the repair loop and a rejected gate sending its producer back run
on - and safe-stop on budget or cycle exhaustion.

**What it deliberately does not implement yet**, so nothing here is mistaken
for more than it is:

- **Dynamic subgraph admission.** `WorkflowGraph.with_subgraph` exists and is
  tested at the graph level, but no `NodeExecutor` outcome here can yet cause
  the running scheduler to admit new nodes mid-run. Wiring a `DECOMPOSE` node
  to do that is Phase 4 work.
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
from ases.kernel.gates import ApprovalDecision, ApprovalProvider, BudgetEntryGate, GateVerdict
from ases.kernel.graph import Edge, EdgeCondition, JoinPolicy, NodeKind, NodeSpec, WorkflowGraph
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
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0


@runtime_checkable
class NodeExecutor(Protocol):
    """Performs one node's work. Stateless with respect to the graph: it is
    handed the node and the current state and returns an outcome - it never
    sees the event store and cannot emit events itself."""

    async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome: ...


#: Statuses the generic readiness scan may pick up directly. Deliberately
#: **excludes** SUCCEEDED and FAILED, even though the state machine permits
#: SUCCEEDED->STALE and FAILED->RETRYING - because both of those predecessor
#: statuses are "sticky": once reached, they never change again on their own,
#: so a join check against a permanently-succeeded (or -failed) predecessor
#: would stay satisfied forever. A generic scan that treated them as
#: re-eligible would restart the node on every subsequent pass for as long as
#: the run stays open for any other reason - not a rare edge case, but the
#: guaranteed outcome for a repair loop or a clarification cycle followed by
#: anything that keeps the run from completing in the same pass. That was
#: exactly the shape of the first bug found while testing this scheduler:
#: `test_linear_run_with_granted_approval_completes` hung outright, and
#: several other tests silently reached the right answer by accident because
#: the run happened to complete before a second, spurious pass could run.
#:
#: The fix: SUCCEEDED and FAILED nodes are moved to STALE/RETRYING **only**
#: by the explicit, one-shot `_propagate_edge_completion` below, called right
#: when a predecessor settles - never inferred by scanning current status.
_DIRECTLY_ELIGIBLE_STATUSES = frozenset(
    {
        NodeStatus.PENDING,
        NodeStatus.REJECTED,
        NodeStatus.RETRYING,
        NodeStatus.STALE,
        NodeStatus.BLOCKED,
    }
)


def _is_directly_eligible(status: NodeStatus) -> bool:
    return status in _DIRECTLY_ELIGIBLE_STATUSES and NodeStatus.READY in LEGAL_TRANSITIONS[status]


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
    ) -> None:
        self.graph = graph
        self.store = store
        self.executors = executors
        self.entry_gate = entry_gate
        self.approvals = approvals
        self.checkpoints = checkpoints
        self.cancel_token = cancel_token or CancelToken()
        self._run_id: UUID | None = None
        self._state: RunState | None = None

    # -- public entry point ---------------------------------------------

    async def run(self, run_id: UUID) -> RunState:
        state = await resume(self.store, run_id, checkpoints=self.checkpoints)
        self._run_id = run_id
        self._state = state

        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.HALTED):
            return state  # resuming a finished run is a no-op, not an error

        try:
            if state.status is RunStatus.CREATED:
                await self._emit(EventType.RUN_STARTED, workflow=self.graph.name)

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
        if state.status is RunStatus.CREATED:
            await self._emit(EventType.RUN_STARTED, workflow=self.graph.name)
        await self._step()
        return state

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
        return any(e.condition is EdgeCondition.ON_FAILURE for e in self.graph.successors(node_id))

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
            elif target_status is NodeStatus.FAILED:
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
            )
            await self._propagate_edge_completion(node.id)
            if not self._has_recovery_edge(node.id):
                await self._emit(
                    EventType.RUN_FAILED,
                    reason=(
                        f"node {node.id!r} failed with no ON_FAILURE recovery edge: {outcome.error}"
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
            )
            await self._emit(
                EventType.ARTIFACT_VALIDATED, node_id=node.id, artifact_hash=artifact_hash
            )

        await self._emit(
            EventType.NODE_EXIT_GATE, node_id=node.id, attempt=attempt, verdict=GateVerdict.PASS
        )

        if node.requires_approval:
            await self._request_approval(node, artifact_hash)
        else:
            await self._emit(EventType.NODE_SUCCEEDED, node_id=node.id, attempt=attempt)
            await self._propagate_edge_completion(node.id)

    # -- completion ---------------------------------------------------------

    def _is_complete(self) -> bool:
        assert self._state is not None
        settled = (NodeStatus.SUCCEEDED, NodeStatus.SKIPPED, NodeStatus.CANCELLED)
        return all(self._state.status_of(n.id) in settled for n in self.graph.nodes)

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
