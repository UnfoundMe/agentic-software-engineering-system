"""The fold: state is derived from events, never stored."""

from __future__ import annotations

from uuid import UUID

import pytest

from ases.kernel.events import Actor, Event, EventType, UnsealedEvent
from ases.kernel.hashing import GENESIS_HASH
from ases.kernel.state import (
    InvalidTransitionError,
    NodeStatus,
    RunStatus,
    fold,
)


def _chain(run_id: UUID, *specs: tuple[EventType, dict[str, object]]) -> list[Event]:
    """Seal a list of (type, kwargs) into a valid chain."""
    events: list[Event] = []
    prev = GENESIS_HASH
    for seq, (event_type, kwargs) in enumerate(specs, start=1):
        node_id = kwargs.pop("node_id", None)
        attempt = kwargs.pop("attempt", None)
        actor = kwargs.pop("actor", Actor.kernel())
        unsealed = UnsealedEvent(
            run_id=run_id,
            type=event_type,
            actor=actor,  # type: ignore[arg-type]
            node_id=node_id,  # type: ignore[arg-type]
            attempt=attempt,  # type: ignore[arg-type]
            payload=kwargs,
        )
        sealed = Event.seal(unsealed, seq=seq, prev_hash=prev)
        prev = sealed.hash
        events.append(sealed)
    return events


def test_empty_log_folds_to_a_created_run(run_id: UUID) -> None:
    state = fold(run_id, [])
    assert state.status is RunStatus.CREATED
    assert state.nodes == {}
    assert state.last_seq == 0


def test_run_lifecycle(run_id: UUID) -> None:
    state = fold(
        run_id,
        _chain(
            run_id,
            (EventType.RUN_CREATED, {"workflow": "greenfield"}),
            (EventType.RUN_STARTED, {}),
            (EventType.RUN_COMPLETED, {}),
        ),
    )
    assert state.status is RunStatus.COMPLETED
    assert state.workflow == "greenfield"
    assert state.ended_at is not None


def test_node_progresses_through_the_machine(run_id: UUID) -> None:
    state = fold(
        run_id,
        _chain(
            run_id,
            (EventType.RUN_CREATED, {}),
            (EventType.NODE_READY, {"node_id": "req"}),
            (EventType.NODE_STARTED, {"node_id": "req", "attempt": 1}),
            (EventType.NODE_SUCCEEDED, {"node_id": "req"}),
        ),
    )
    node = state.nodes["req"]
    assert node.status is NodeStatus.SUCCEEDED
    assert node.attempt == 1
    assert node.started_at is not None and node.ended_at is not None


def test_illegal_transition_is_refused(run_id: UUID) -> None:
    """A log that cannot have happened stops the run loudly.

    Smoothing this over would produce a plausible-looking state that does not
    correspond to anything the system actually did.
    """
    events = _chain(
        run_id,
        (EventType.RUN_CREATED, {}),
        (EventType.NODE_SUCCEEDED, {"node_id": "req"}),  # never ran
    )
    with pytest.raises(InvalidTransitionError) as excinfo:
        fold(run_id, events)
    assert excinfo.value.node_id == "req"


def test_repeating_an_event_is_idempotent(run_id: UUID) -> None:
    """Replay must be safe; the fold is applied to the same log repeatedly."""
    state = fold(
        run_id,
        _chain(
            run_id,
            (EventType.NODE_READY, {"node_id": "a"}),
            (EventType.NODE_READY, {"node_id": "a"}),
        ),
    )
    assert state.nodes["a"].status is NodeStatus.READY


def test_artifact_is_recorded_with_its_lineage(run_id: UUID) -> None:
    state = fold(
        run_id,
        _chain(
            run_id,
            (EventType.NODE_READY, {"node_id": "arch"}),
            (EventType.NODE_STARTED, {"node_id": "arch"}),
            (
                EventType.ARTIFACT_PRODUCED,
                {
                    "node_id": "arch",
                    "artifact_hash": "h-design",
                    "kind": "DesignSpec",
                    "inputs": ["h-req"],
                },
            ),
        ),
    )
    record = state.artifacts["h-design"]
    assert record.kind == "DesignSpec"
    assert record.node_id == "arch"
    assert record.inputs == ("h-req",)
    assert state.nodes["arch"].produced == ("h-design",)


def test_artifact_content_is_captured_when_the_payload_carries_it(run_id: UUID) -> None:
    state = fold(
        run_id,
        _chain(
            run_id,
            (EventType.NODE_READY, {"node_id": "arch"}),
            (EventType.NODE_STARTED, {"node_id": "arch"}),
            (
                EventType.ARTIFACT_PRODUCED,
                {
                    "node_id": "arch",
                    "artifact_hash": "h-design",
                    "kind": "DesignSpec",
                    "content": {"summary": "a design"},
                },
            ),
        ),
    )
    assert state.artifact_content["h-design"] == {"summary": "a design"}


def test_artifact_content_is_absent_without_error_when_the_payload_has_none(run_id: UUID) -> None:
    """An older, content-less export (recorded before this field existed)
    must replay cleanly - a missing `content` key is not a fold error."""
    state = fold(
        run_id,
        _chain(
            run_id,
            (EventType.NODE_READY, {"node_id": "arch"}),
            (EventType.NODE_STARTED, {"node_id": "arch"}),
            (
                EventType.ARTIFACT_PRODUCED,
                {"node_id": "arch", "artifact_hash": "h-design", "kind": "DesignSpec"},
            ),
        ),
    )
    assert "h-design" not in state.artifact_content
    assert state.artifacts["h-design"].kind == "DesignSpec"


def test_approval_binds_to_the_artifact_hash(run_id: UUID) -> None:
    """Binding is what makes revocation correct rather than advisory."""
    state = fold(
        run_id,
        _chain(
            run_id,
            (EventType.NODE_READY, {"node_id": "gate1"}),
            (EventType.NODE_STARTED, {"node_id": "gate1"}),
            (EventType.APPROVAL_REQUESTED, {"node_id": "gate1", "artifact_hash": "h-req"}),
            (
                EventType.APPROVAL_GRANTED,
                {"node_id": "gate1", "artifact_hash": "h-req", "actor": Actor.human("alice")},
            ),
        ),
    )
    approval = state.approvals["gate1"]
    assert approval.granted
    assert approval.artifact_hash == "h-req"
    assert state.nodes["gate1"].status is NodeStatus.SUCCEEDED


def test_revocation_clears_the_grant(run_id: UUID) -> None:
    """An upstream change must not leave an approval that still reads as granted."""
    state = fold(
        run_id,
        _chain(
            run_id,
            (EventType.NODE_READY, {"node_id": "gate1"}),
            (EventType.NODE_STARTED, {"node_id": "gate1"}),
            (EventType.APPROVAL_REQUESTED, {"node_id": "gate1", "artifact_hash": "h1"}),
            (EventType.APPROVAL_GRANTED, {"node_id": "gate1", "artifact_hash": "h1"}),
            (EventType.APPROVAL_REVOKED, {"node_id": "gate1", "reason": "upstream changed"}),
        ),
    )
    approval = state.approvals["gate1"]
    assert approval.revoked and not approval.granted
    assert approval.revoked_reason == "upstream changed"


def test_rejected_gate_sends_the_node_back_rather_than_failing_the_run(run_id: UUID) -> None:
    """The clarification cycle: a human rejection is a backward edge, not an error."""
    state = fold(
        run_id,
        _chain(
            run_id,
            (EventType.NODE_READY, {"node_id": "gate1"}),
            (EventType.NODE_STARTED, {"node_id": "gate1"}),
            (EventType.APPROVAL_REQUESTED, {"node_id": "gate1", "artifact_hash": "h1"}),
            (EventType.APPROVAL_REJECTED, {"node_id": "gate1", "reason": "too vague"}),
            (EventType.NODE_READY, {"node_id": "gate1"}),
        ),
    )
    assert state.nodes["gate1"].status is NodeStatus.READY
    assert state.status is not RunStatus.FAILED


def test_stale_drops_the_produced_artifacts(run_id: UUID) -> None:
    """A stale node's output must not be mistaken for current."""
    state = fold(
        run_id,
        _chain(
            run_id,
            (EventType.NODE_READY, {"node_id": "arch"}),
            (EventType.NODE_STARTED, {"node_id": "arch"}),
            (EventType.ARTIFACT_PRODUCED, {"node_id": "arch", "artifact_hash": "h1"}),
            (EventType.NODE_SUCCEEDED, {"node_id": "arch"}),
            (EventType.NODE_MARKED_STALE, {"node_id": "arch"}),
        ),
    )
    assert state.nodes["arch"].status is NodeStatus.STALE
    assert state.nodes["arch"].produced == ()


def test_usage_and_violations_accumulate(run_id: UUID) -> None:
    state = fold(
        run_id,
        _chain(
            run_id,
            (EventType.LLM_COMPLETED, {"input_tokens": 100, "output_tokens": 20, "usd": 0.5}),
            (EventType.LLM_COMPLETED, {"input_tokens": 50, "output_tokens": 10, "usd": 0.25}),
            (EventType.POLICY_VIOLATION, {"rule": "no_raw_sql_migration"}),
        ),
    )
    assert state.usage.total_tokens == 180
    assert state.usage.usd == pytest.approx(0.75)
    assert state.policy_violations == 1


def test_repair_cycles_are_counted(run_id: UUID) -> None:
    """The cycle counter is what the safe-stop control reads."""
    state = fold(
        run_id,
        _chain(
            run_id,
            (EventType.NODE_READY, {"node_id": "test"}),
            (EventType.NODE_STARTED, {"node_id": "test"}),
            (EventType.NODE_VALIDATING, {"node_id": "test"}),
            (EventType.NODE_REPAIRING, {"node_id": "test"}),
            (EventType.NODE_STARTED, {"node_id": "test"}),
            (EventType.NODE_VALIDATING, {"node_id": "test"}),
            (EventType.NODE_REPAIRING, {"node_id": "test"}),
        ),
    )
    assert state.nodes["test"].cycle_count == 2


def test_fold_is_deterministic(run_id: UUID) -> None:
    """Two folds of the same log must agree - the dashboard depends on it."""
    events = _chain(
        run_id,
        (EventType.RUN_CREATED, {"workflow": "greenfield"}),
        (EventType.NODE_READY, {"node_id": "a"}),
        (EventType.NODE_STARTED, {"node_id": "a"}),
        (EventType.NODE_SUCCEEDED, {"node_id": "a"}),
    )
    assert fold(run_id, events).model_dump() == fold(run_id, events).model_dump()
