"""Event sealing and chain integrity."""

from __future__ import annotations

from uuid import uuid4

from ases.kernel.events import Actor, ActorKind, Event, EventType, UnsealedEvent
from ases.kernel.hashing import GENESIS_HASH
from ases.kernel.state import HANDLED_EVENT_TYPES


def _unsealed(**kw: object) -> UnsealedEvent:
    return UnsealedEvent(
        run_id=uuid4(),
        type=EventType.RUN_CREATED,
        actor=Actor.kernel(),
        **kw,  # type: ignore[arg-type]
    )


def test_seal_produces_a_recomputable_hash() -> None:
    event = Event.seal(_unsealed(), seq=1, prev_hash=GENESIS_HASH)
    assert event.recompute_hash() == event.hash


def test_first_event_is_genesis() -> None:
    assert Event.seal(_unsealed(), seq=1, prev_hash=GENESIS_HASH).is_genesis


def test_altering_payload_breaks_the_hash() -> None:
    """The whole tamper-evidence claim reduces to this."""
    event = Event.seal(_unsealed(), seq=1, prev_hash=GENESIS_HASH)
    forged = event.model_copy(update={"payload": {"tampered": True}})
    assert forged.recompute_hash() != forged.hash


def test_altering_the_actor_breaks_the_hash() -> None:
    """ "Who approved this" must be as protected as "what was approved"."""
    event = Event.seal(_unsealed(), seq=1, prev_hash=GENESIS_HASH)
    forged = event.model_copy(update={"actor": Actor.human("someone-else")})
    assert forged.recompute_hash() != forged.hash


def test_reordering_breaks_the_link() -> None:
    a = Event.seal(_unsealed(), seq=1, prev_hash=GENESIS_HASH)
    b = Event.seal(_unsealed(), seq=2, prev_hash=a.hash)
    # b's own digest stays valid; it is the *link* that detects reordering.
    assert b.recompute_hash() == b.hash
    assert b.prev_hash != GENESIS_HASH


def test_events_are_immutable() -> None:
    event = Event.seal(_unsealed(), seq=1, prev_hash=GENESIS_HASH)
    try:
        event.seq = 99  # type: ignore[misc]
    except (ValueError, AttributeError, TypeError):
        return
    raise AssertionError("Event should be frozen")


def test_actor_renders_readably() -> None:
    assert str(Actor.human("alice")) == "human:alice"
    assert str(Actor.kernel()) == ActorKind.KERNEL


def test_fold_covers_every_event_type() -> None:
    """A new EventType must be given a fold case, not silently ignored.

    This is the guard for the one change that is deliberately hard: adding an
    event type is the only edit that necessarily touches the kernel.
    """
    missing = set(EventType) - HANDLED_EVENT_TYPES
    assert not missing, f"EventType(s) with no fold case: {sorted(m.value for m in missing)}"
