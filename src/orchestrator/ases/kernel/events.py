"""The event model.

The event log is the system's only source of truth. Run state is a fold over
it (`kernel.state.fold`); checkpoints are a disposable cache. Audit trail,
decision lineage and reliability metrics are all *derived* from this log rather
than instrumented separately, which is why they cannot drift from what actually
happened.

Each event carries `prev_hash`, the digest of its predecessor, forming a chain
that makes retroactive edits detectable. The chain is only meaningful if it is
checked, so `EventStore.verify_chain` recomputes it end to end.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from ases.kernel.hashing import GENESIS_HASH, HASH_ALGORITHM_VERSION, digest


def utc_now() -> datetime:
    return datetime.now(UTC)


class EventType(StrEnum):
    """Every state transition in the system.

    Adding a member is the one change that necessarily touches the kernel -
    a new type needs a fold case. Everything else in the system is a plug-in.
    """

    # --- run lifecycle ---------------------------------------------------
    RUN_CREATED = "run.created"
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_HALTED = "run.halted"  # safe-stop: budget, cycle or policy exhausted

    # --- node lifecycle --------------------------------------------------
    NODE_READY = "node.ready"
    NODE_BLOCKED = "node.blocked"
    NODE_ENTRY_GATE = "node.entry_gate"  # payload.verdict: pass|fail|escalate
    NODE_STARTED = "node.started"
    NODE_VALIDATING = "node.validating"
    NODE_REPAIRING = "node.repairing"
    NODE_EXIT_GATE = "node.exit_gate"
    NODE_SUCCEEDED = "node.succeeded"
    NODE_FAILED = "node.failed"
    NODE_RETRY_SCHEDULED = "node.retry_scheduled"
    NODE_FALLBACK_TAKEN = "node.fallback_taken"
    NODE_CANCELLED = "node.cancelled"
    NODE_SKIPPED = "node.skipped"
    NODE_MARKED_STALE = "node.marked_stale"  # an upstream artifact changed
    NODE_COMPENSATING = "node.compensating"
    NODE_ROLLED_BACK = "node.rolled_back"

    # --- artifacts and lineage -------------------------------------------
    ARTIFACT_PRODUCED = "artifact.produced"
    ARTIFACT_VALIDATED = "artifact.validated"
    ARTIFACT_REJECTED = "artifact.rejected"

    # --- dynamic subgraph admission --------------------------------------
    SUBGRAPH_PROPOSED = "subgraph.proposed"
    SUBGRAPH_ADMITTED = "subgraph.admitted"
    SUBGRAPH_REJECTED = "subgraph.rejected"

    # --- human approval ---------------------------------------------------
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_REJECTED = "approval.rejected"
    APPROVAL_REVOKED = "approval.revoked"  # bound artifact hash changed

    # --- governance -------------------------------------------------------
    POLICY_EVALUATED = "policy.evaluated"
    POLICY_VIOLATION = "policy.violation"

    # --- tools ------------------------------------------------------------
    TOOL_INVOKED = "tool.invoked"
    TOOL_SUCCEEDED = "tool.succeeded"
    TOOL_FAILED = "tool.failed"
    TOOL_COMPENSATED = "tool.compensated"

    # --- LLM boundary -----------------------------------------------------
    LLM_REQUESTED = "llm.requested"
    LLM_COMPLETED = "llm.completed"  # payload carries tokens, cost, model

    # --- re-planning ------------------------------------------------------
    REPLAN_TRIGGERED = "replan.triggered"
    REPLAN_COMPUTED = "replan.computed"

    # --- bookkeeping ------------------------------------------------------
    CHECKPOINT_WRITTEN = "checkpoint.written"


class ActorKind(StrEnum):
    KERNEL = "kernel"
    AGENT = "agent"
    HUMAN = "human"
    TOOL = "tool"
    SYSTEM = "system"


class Actor(BaseModel):
    """Who caused an event.

    Recorded on every event because "an approval was granted" is not an audit
    record until it says by whom.
    """

    model_config = ConfigDict(frozen=True)

    kind: ActorKind
    id: str | None = None

    def __str__(self) -> str:
        return f"{self.kind}:{self.id}" if self.id else str(self.kind)

    @classmethod
    def kernel(cls) -> Actor:
        return cls(kind=ActorKind.KERNEL)

    @classmethod
    def agent(cls, name: str) -> Actor:
        return cls(kind=ActorKind.AGENT, id=name)

    @classmethod
    def human(cls, identity: str) -> Actor:
        return cls(kind=ActorKind.HUMAN, id=identity)

    @classmethod
    def tool(cls, name: str) -> Actor:
        return cls(kind=ActorKind.TOOL, id=name)


class UnsealedEvent(BaseModel):
    """An event before the store assigns it a sequence number and hash.

    Callers construct these; only the store may produce a sealed `Event`. The
    split exists so `seq` and `prev_hash` cannot be forged by a caller - they
    are assigned under the store's per-run write lock.
    """

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    type: EventType
    actor: Actor
    payload: Mapping[str, Any] = Field(default_factory=dict)
    node_id: str | None = None
    attempt: int | None = None
    created_at: datetime = Field(default_factory=utc_now)


class Event(BaseModel):
    """A sealed, chained event. Immutable."""

    model_config = ConfigDict(frozen=True)

    seq: int
    event_id: UUID
    run_id: UUID
    type: EventType
    actor: Actor
    payload: Mapping[str, Any]
    node_id: str | None
    attempt: int | None
    created_at: datetime
    prev_hash: str
    hash: str

    # ------------------------------------------------------------------

    @staticmethod
    def _hashable(
        *,
        seq: int,
        event_id: UUID,
        run_id: UUID,
        type_: EventType,
        actor: Actor,
        payload: Mapping[str, Any],
        node_id: str | None,
        attempt: int | None,
        created_at: datetime,
        prev_hash: str,
    ) -> dict[str, Any]:
        """The exact set of fields covered by the digest.

        `algorithm` is included so a future change to the hashing scheme
        produces visibly different digests rather than silently colliding
        with the old ones.
        """
        return {
            "algorithm": HASH_ALGORITHM_VERSION,
            "seq": seq,
            "event_id": event_id,
            "run_id": run_id,
            "type": type_,
            "actor": {"kind": actor.kind, "id": actor.id},
            "payload": payload,
            "node_id": node_id,
            "attempt": attempt,
            "created_at": created_at,
            "prev_hash": prev_hash,
        }

    @classmethod
    def seal(
        cls,
        unsealed: UnsealedEvent,
        *,
        seq: int,
        prev_hash: str,
        event_id: UUID | None = None,
    ) -> Self:
        """Assign position in the chain and compute the digest.

        Only an `EventStore` should call this, holding the run's write lock.
        """
        eid = event_id or uuid4()
        body = cls._hashable(
            seq=seq,
            event_id=eid,
            run_id=unsealed.run_id,
            type_=unsealed.type,
            actor=unsealed.actor,
            payload=unsealed.payload,
            node_id=unsealed.node_id,
            attempt=unsealed.attempt,
            created_at=unsealed.created_at,
            prev_hash=prev_hash,
        )
        return cls(
            seq=seq,
            event_id=eid,
            run_id=unsealed.run_id,
            type=unsealed.type,
            actor=unsealed.actor,
            payload=unsealed.payload,
            node_id=unsealed.node_id,
            attempt=unsealed.attempt,
            created_at=unsealed.created_at,
            prev_hash=prev_hash,
            hash=digest(body),
        )

    def recompute_hash(self) -> str:
        """Recompute this event's digest from its own fields.

        Equality with `self.hash` means the event has not been altered since it
        was sealed.
        """
        return digest(
            self._hashable(
                seq=self.seq,
                event_id=self.event_id,
                run_id=self.run_id,
                type_=self.type,
                actor=self.actor,
                payload=self.payload,
                node_id=self.node_id,
                attempt=self.attempt,
                created_at=self.created_at,
                prev_hash=self.prev_hash,
            )
        )

    @property
    def is_genesis(self) -> bool:
        return self.prev_hash == GENESIS_HASH


class ChainProblem(BaseModel):
    model_config = ConfigDict(frozen=True)

    seq: int
    reason: str


class ChainVerification(BaseModel):
    """Result of recomputing a run's hash chain."""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    events_checked: int
    problems: tuple[ChainProblem, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.problems
