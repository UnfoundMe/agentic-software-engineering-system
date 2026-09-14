"""Run state as a fold over the event log.

There is no mutable run state of record. `RunState = fold(events)`. Everything
the scheduler, the dashboard and the metrics layer read is produced by this
function, which is why they cannot disagree about what happened.

The fold is **total and strict**: it applies a transition table and raises on an
illegal transition. An illegal transition in a persisted log means the log is
corrupt or the kernel has a bug, and both should stop the run loudly rather than
be smoothed over into a plausible-looking state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ases.kernel.events import Event, EventType


class InvalidTransitionError(RuntimeError):
    """A node was moved between states the machine does not permit."""

    def __init__(self, node_id: str, current: NodeStatus, attempted: NodeStatus, seq: int) -> None:
        super().__init__(
            f"node {node_id!r} cannot move {current} -> {attempted} (event seq {seq}). "
            "The event log is inconsistent with the node state machine."
        )
        self.node_id = node_id
        self.current = current
        self.attempted = attempted
        self.seq = seq


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    HALTED = "halted"  # safe-stop


class NodeStatus(StrEnum):
    PENDING = "pending"
    BLOCKED = "blocked"
    READY = "ready"
    ENTRY_GATE = "entry_gate"
    RUNNING = "running"
    VALIDATING = "validating"
    REPAIRING = "repairing"
    EXIT_GATE = "exit_gate"
    AWAITING_APPROVAL = "awaiting_approval"
    REJECTED = "rejected"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRYING = "retrying"
    FALLBACK = "fallback"
    COMPENSATING = "compensating"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    STALE = "stale"
    HALTED = "halted"


TERMINAL_STATUSES: frozenset[NodeStatus] = frozenset(
    {NodeStatus.CANCELLED, NodeStatus.SKIPPED, NodeStatus.HALTED}
)

#: Legal node transitions. Kept explicit rather than derived so that the
#: permitted shape of a run is reviewable in one place.
LEGAL_TRANSITIONS: Mapping[NodeStatus, frozenset[NodeStatus]] = {
    NodeStatus.PENDING: frozenset(
        {NodeStatus.READY, NodeStatus.BLOCKED, NodeStatus.SKIPPED, NodeStatus.CANCELLED}
    ),
    NodeStatus.BLOCKED: frozenset(
        {NodeStatus.READY, NodeStatus.SKIPPED, NodeStatus.CANCELLED, NodeStatus.HALTED}
    ),
    NodeStatus.READY: frozenset(
        {
            NodeStatus.ENTRY_GATE,
            # A node with no configured entry gates goes straight to RUNNING.
            # Requiring a gate event for every node would add an event per node
            # that records no decision.
            NodeStatus.RUNNING,
            NodeStatus.BLOCKED,
            NodeStatus.SKIPPED,
            NodeStatus.CANCELLED,
            NodeStatus.HALTED,
        }
    ),
    NodeStatus.ENTRY_GATE: frozenset(
        {
            NodeStatus.RUNNING,
            NodeStatus.FAILED,
            NodeStatus.AWAITING_APPROVAL,
            NodeStatus.CANCELLED,
            NodeStatus.HALTED,
        }
    ),
    NodeStatus.RUNNING: frozenset(
        {
            NodeStatus.VALIDATING,
            NodeStatus.EXIT_GATE,
            # A gate node evaluates and then asks for a human decision without
            # a separate exit-gate event.
            NodeStatus.AWAITING_APPROVAL,
            NodeStatus.SUCCEEDED,
            NodeStatus.FAILED,
            NodeStatus.CANCELLED,
            NodeStatus.HALTED,
        }
    ),
    NodeStatus.VALIDATING: frozenset(
        {
            NodeStatus.EXIT_GATE,
            NodeStatus.REPAIRING,
            NodeStatus.SUCCEEDED,
            NodeStatus.FAILED,
            NodeStatus.CANCELLED,
        }
    ),
    NodeStatus.REPAIRING: frozenset(
        {NodeStatus.RUNNING, NodeStatus.VALIDATING, NodeStatus.FAILED, NodeStatus.CANCELLED}
    ),
    NodeStatus.EXIT_GATE: frozenset(
        {
            NodeStatus.SUCCEEDED,
            NodeStatus.FAILED,
            NodeStatus.AWAITING_APPROVAL,
            NodeStatus.CANCELLED,
        }
    ),
    NodeStatus.AWAITING_APPROVAL: frozenset(
        {NodeStatus.SUCCEEDED, NodeStatus.REJECTED, NodeStatus.CANCELLED, NodeStatus.HALTED}
    ),
    # A rejected gate does not fail the run: it sends the producing node back
    # to be redone with the human's clarification. That backward edge is the
    # clarification cycle, and it is bounded by the node's cycle budget.
    # RETRYING is the one-shot re-staging target fired by
    # `Scheduler._propagate_edge_completion` when that producer settles
    # again - see that method's docstring for why a REJECTED node must not
    # be picked up by the generic readiness scan on its own.
    NodeStatus.REJECTED: frozenset(
        {NodeStatus.READY, NodeStatus.RETRYING, NodeStatus.FAILED, NodeStatus.HALTED}
    ),
    NodeStatus.SUCCEEDED: frozenset({NodeStatus.STALE}),
    NodeStatus.FAILED: frozenset(
        {
            NodeStatus.RETRYING,
            NodeStatus.FALLBACK,
            NodeStatus.COMPENSATING,
            NodeStatus.HALTED,
        }
    ),
    NodeStatus.RETRYING: frozenset(
        {NodeStatus.READY, NodeStatus.ENTRY_GATE, NodeStatus.CANCELLED, NodeStatus.HALTED}
    ),
    NodeStatus.FALLBACK: frozenset({NodeStatus.SUCCEEDED, NodeStatus.FAILED, NodeStatus.HALTED}),
    NodeStatus.COMPENSATING: frozenset({NodeStatus.ROLLED_BACK, NodeStatus.FAILED}),
    NodeStatus.ROLLED_BACK: frozenset({NodeStatus.READY, NodeStatus.HALTED}),
    # An upstream artifact changed. The node must run again, and any approval
    # bound to its output has already been revoked by the replanner.
    NodeStatus.STALE: frozenset({NodeStatus.PENDING, NodeStatus.READY, NodeStatus.SKIPPED}),
    NodeStatus.CANCELLED: frozenset(),
    NodeStatus.SKIPPED: frozenset(),
    NodeStatus.HALTED: frozenset(),
}


class Usage(BaseModel):
    """Consumption accumulated across a run, for budget enforcement."""

    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ArtifactRecord(BaseModel):
    """Provenance of one artifact. The lineage DAG is built from these."""

    model_config = ConfigDict(frozen=True)

    artifact_hash: str
    kind: str
    node_id: str
    produced_at: datetime
    inputs: tuple[str, ...] = ()
    validated: bool = False


class ApprovalRecord(BaseModel):
    """A human decision, bound to the artifact hash it was granted against.

    `artifact_hash` is what makes revocation correct: when re-planning changes
    the artifact, the approval no longer refers to anything that exists, and
    inheriting it would mean approving content nobody reviewed.
    """

    node_id: str
    artifact_hash: str
    granted: bool
    actor: str
    decided_at: datetime
    reason: str | None = None
    revoked: bool = False
    revoked_reason: str | None = None


class NodeState(BaseModel):
    node_id: str
    status: NodeStatus = NodeStatus.PENDING
    attempt: int = 0
    cycle_count: int = 0
    started_at: datetime | None = None
    ended_at: datetime | None = None
    last_error: str | None = None
    produced: tuple[str, ...] = ()

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


class RunState(BaseModel):
    """A derived snapshot. Produced only by `fold`; never mutated by callers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: UUID
    status: RunStatus = RunStatus.CREATED
    workflow: str = ""
    last_seq: int = 0
    created_at: datetime | None = None
    ended_at: datetime | None = None
    halt_reason: str | None = None

    nodes: dict[str, NodeState] = Field(default_factory=dict)
    artifacts: dict[str, ArtifactRecord] = Field(default_factory=dict)
    #: Full artifact content, keyed by hash - present only for events that
    #: carried a `content` field in their payload (see `ARTIFACT_PRODUCED`
    #: below). Kept separate from `ArtifactRecord`, which is pure provenance
    #: metadata and predates this field; a run replayed from an older,
    #: content-less export simply has an empty dict here, never a KeyError.
    artifact_content: dict[str, Mapping[str, Any]] = Field(default_factory=dict)
    approvals: dict[str, ApprovalRecord] = Field(default_factory=dict)
    #: Every subgraph admitted during this run, in admission order, each a
    #: `{"nodes": [...], "edges": [...]}` record copied verbatim from its
    #: `SUBGRAPH_ADMITTED` payload.
    #:
    #: Held here because the event log is the only source of truth for what a
    #: run actually executed (CLAUDE.md section 4), and a dynamically admitted
    #: implementation subgraph is part of that. Without it, resuming a run
    #: that got past `decompose` would rebuild the *static* YAML graph and
    #: then fold a state referring to dozens of nodes that graph has never
    #: heard of - `Scheduler._readmit_recorded_subgraphs` replays these
    #: instead, through the same `with_subgraph` validation the live
    #: admission used.
    admitted_subgraphs: list[Mapping[str, Any]] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    policy_violations: int = 0

    # -- queries used by the scheduler ---------------------------------

    def node(self, node_id: str) -> NodeState:
        return self.nodes.setdefault(node_id, NodeState(node_id=node_id))

    def status_of(self, node_id: str) -> NodeStatus:
        state = self.nodes.get(node_id)
        return state.status if state else NodeStatus.PENDING

    def in_status(self, *statuses: NodeStatus) -> tuple[str, ...]:
        wanted = set(statuses)
        return tuple(nid for nid, s in self.nodes.items() if s.status in wanted)

    @property
    def open_approvals(self) -> tuple[ApprovalRecord, ...]:
        return tuple(a for a in self.approvals.values() if not a.granted and not a.revoked)


# ---------------------------------------------------------------------------
# The fold
# ---------------------------------------------------------------------------

#: Event types that carry information but do not move the state machine.
#: Listed explicitly so that adding a new EventType without deciding how it
#: folds is a test failure rather than a silent no-op.
_INFORMATIONAL: frozenset[EventType] = frozenset(
    {
        EventType.POLICY_EVALUATED,
        EventType.TOOL_INVOKED,
        EventType.TOOL_SUCCEEDED,
        EventType.TOOL_FAILED,
        EventType.TOOL_COMPENSATED,
        EventType.LLM_REQUESTED,
        EventType.SUBGRAPH_PROPOSED,
        EventType.SUBGRAPH_REJECTED,
        EventType.REPLAN_TRIGGERED,
        EventType.REPLAN_COMPUTED,
        EventType.CHECKPOINT_WRITTEN,
        EventType.ARTIFACT_REJECTED,
    }
)

_NODE_STATUS_EVENTS: Mapping[EventType, NodeStatus] = {
    EventType.NODE_READY: NodeStatus.READY,
    EventType.NODE_BLOCKED: NodeStatus.BLOCKED,
    # The gates are states the node occupies, not just annotations: the
    # dashboard must be able to show "waiting at the entry gate", and a run
    # that stalls there is a different diagnosis from one stalled in RUNNING.
    EventType.NODE_ENTRY_GATE: NodeStatus.ENTRY_GATE,
    EventType.NODE_EXIT_GATE: NodeStatus.EXIT_GATE,
    EventType.NODE_STARTED: NodeStatus.RUNNING,
    EventType.NODE_VALIDATING: NodeStatus.VALIDATING,
    EventType.NODE_REPAIRING: NodeStatus.REPAIRING,
    EventType.NODE_SUCCEEDED: NodeStatus.SUCCEEDED,
    EventType.NODE_FAILED: NodeStatus.FAILED,
    EventType.NODE_RETRY_SCHEDULED: NodeStatus.RETRYING,
    EventType.NODE_FALLBACK_TAKEN: NodeStatus.FALLBACK,
    EventType.NODE_CANCELLED: NodeStatus.CANCELLED,
    EventType.NODE_SKIPPED: NodeStatus.SKIPPED,
    EventType.NODE_MARKED_STALE: NodeStatus.STALE,
    EventType.NODE_COMPENSATING: NodeStatus.COMPENSATING,
    EventType.NODE_ROLLED_BACK: NodeStatus.ROLLED_BACK,
}

_RUN_STATUS_EVENTS: Mapping[EventType, RunStatus] = {
    EventType.RUN_CREATED: RunStatus.CREATED,
    EventType.RUN_STARTED: RunStatus.RUNNING,
    EventType.RUN_COMPLETED: RunStatus.COMPLETED,
    EventType.RUN_FAILED: RunStatus.FAILED,
    EventType.RUN_HALTED: RunStatus.HALTED,
}

#: Event types handled by the `match` block below, as opposed to the two
#: lookup tables. Declared rather than inferred so that
#: `test_fold_covers_every_event_type` can prove the union is exhaustive: a new
#: EventType that nobody decided how to fold fails the build.
_EXPLICITLY_HANDLED: frozenset[EventType] = frozenset(
    {
        EventType.ARTIFACT_PRODUCED,
        EventType.ARTIFACT_VALIDATED,
        EventType.SUBGRAPH_ADMITTED,
        EventType.APPROVAL_REQUESTED,
        EventType.APPROVAL_GRANTED,
        EventType.APPROVAL_REJECTED,
        EventType.APPROVAL_REVOKED,
        EventType.POLICY_VIOLATION,
        EventType.LLM_COMPLETED,
    }
)

HANDLED_EVENT_TYPES: frozenset[EventType] = (
    frozenset(_RUN_STATUS_EVENTS)
    | frozenset(_NODE_STATUS_EVENTS)
    | _EXPLICITLY_HANDLED
    | _INFORMATIONAL
)


def _str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _transition(state: RunState, node_id: str, to: NodeStatus, event: Event) -> NodeState:
    node = state.node(node_id)
    if node.status == to:
        return node  # idempotent re-assertion, e.g. replay of the same event
    allowed = LEGAL_TRANSITIONS[node.status]
    if to not in allowed:
        raise InvalidTransitionError(node_id, node.status, to, event.seq)
    node.status = to
    return node


def apply(state: RunState, event: Event) -> RunState:
    """Apply one event. Mutates and returns `state` for fold efficiency."""
    state.last_seq = event.seq
    payload = event.payload

    if event.type in _RUN_STATUS_EVENTS:
        state.status = _RUN_STATUS_EVENTS[event.type]
        if event.type == EventType.RUN_CREATED:
            state.created_at = event.created_at
            state.workflow = _str(payload, "workflow") or ""
        elif event.type in (EventType.RUN_COMPLETED, EventType.RUN_FAILED, EventType.RUN_HALTED):
            state.ended_at = event.created_at
            state.halt_reason = _str(payload, "reason")
        return state

    if event.type in _NODE_STATUS_EVENTS:
        if event.node_id is None:
            raise ValueError(f"{event.type} at seq {event.seq} has no node_id")
        node = _transition(state, event.node_id, _NODE_STATUS_EVENTS[event.type], event)
        if event.attempt is not None:
            node.attempt = event.attempt
        if event.type == EventType.NODE_STARTED and node.started_at is None:
            node.started_at = event.created_at
        if event.type in (EventType.NODE_SUCCEEDED, EventType.NODE_FAILED):
            node.ended_at = event.created_at
        if event.type == EventType.NODE_FAILED:
            node.last_error = _str(payload, "error")
        if event.type == EventType.NODE_REPAIRING:
            node.cycle_count += 1
        if event.type == EventType.NODE_MARKED_STALE:
            # Output is no longer trustworthy; drop the association so a stale
            # artifact cannot be mistaken for a current one.
            node.produced = ()
        return state

    match event.type:
        case EventType.ARTIFACT_PRODUCED:
            artifact_hash = _str(payload, "artifact_hash")
            if artifact_hash is None:
                raise ValueError(f"artifact.produced at seq {event.seq} has no artifact_hash")
            inputs = payload.get("inputs") or ()
            state.artifacts[artifact_hash] = ArtifactRecord(
                artifact_hash=artifact_hash,
                kind=_str(payload, "kind") or "unknown",
                node_id=event.node_id or "",
                produced_at=event.created_at,
                inputs=tuple(str(i) for i in inputs),
            )
            content = payload.get("content")
            if isinstance(content, Mapping):
                state.artifact_content[artifact_hash] = content
            if event.node_id:
                node = state.node(event.node_id)
                node.produced = (*node.produced, artifact_hash)

        case EventType.ARTIFACT_VALIDATED:
            artifact_hash = _str(payload, "artifact_hash")
            record = state.artifacts.get(artifact_hash or "")
            if record is not None:
                state.artifacts[record.artifact_hash] = record.model_copy(
                    update={"validated": True}
                )

        case EventType.SUBGRAPH_ADMITTED:
            for nid in payload.get("node_ids", ()):
                state.node(str(nid))
            nodes = payload.get("nodes")
            edges = payload.get("edges")
            if nodes:
                # Recorded whole, not just by id: `Scheduler` rebuilds the
                # admitted graph from exactly this on resume. An older event
                # that carried only `node_ids` folds as before and simply
                # contributes no rebuildable record.
                state.admitted_subgraphs.append({"nodes": nodes, "edges": edges or []})

        case EventType.APPROVAL_REQUESTED:
            if event.node_id is None:
                raise ValueError(f"approval.requested at seq {event.seq} has no node_id")
            _transition(state, event.node_id, NodeStatus.AWAITING_APPROVAL, event)
            state.approvals[event.node_id] = ApprovalRecord(
                node_id=event.node_id,
                artifact_hash=_str(payload, "artifact_hash") or "",
                granted=False,
                actor="",
                decided_at=event.created_at,
            )

        case EventType.APPROVAL_GRANTED | EventType.APPROVAL_REJECTED:
            if event.node_id is None:
                raise ValueError(f"{event.type} at seq {event.seq} has no node_id")
            granted = event.type == EventType.APPROVAL_GRANTED
            _transition(
                state,
                event.node_id,
                NodeStatus.SUCCEEDED if granted else NodeStatus.REJECTED,
                event,
            )
            state.approvals[event.node_id] = ApprovalRecord(
                node_id=event.node_id,
                artifact_hash=_str(payload, "artifact_hash") or "",
                granted=granted,
                actor=str(event.actor),
                decided_at=event.created_at,
                reason=_str(payload, "reason"),
            )

        case EventType.APPROVAL_REVOKED:
            if event.node_id is None:
                raise ValueError(f"approval.revoked at seq {event.seq} has no node_id")
            existing = state.approvals.get(event.node_id)
            if existing is not None:
                existing.revoked = True
                existing.granted = False
                existing.revoked_reason = _str(payload, "reason")

        case EventType.POLICY_VIOLATION:
            state.policy_violations += 1

        case EventType.LLM_COMPLETED:
            state.usage.input_tokens += int(payload.get("input_tokens", 0) or 0)
            state.usage.output_tokens += int(payload.get("output_tokens", 0) or 0)
            state.usage.usd += float(payload.get("usd", 0.0) or 0.0)

        case _ if event.type in _INFORMATIONAL:
            pass

        case _:  # pragma: no cover - guarded by test_fold_handles_every_event_type
            raise NotImplementedError(
                f"{event.type} has no fold case. Every EventType must either move "
                "the state machine or be listed in _INFORMATIONAL."
            )

    return state


def fold(run_id: UUID, events: Iterable[Event]) -> RunState:
    """Build run state from an event stream. The only way `RunState` is made."""
    state = RunState(run_id=run_id)
    for event in events:
        apply(state, event)
    return state
