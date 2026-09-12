"""Event store protocol and shared chain verification.

Two implementations exist:

- `PostgresEventStore` - the source of truth. Append-only is enforced by the
  database (privileges plus a trigger), not merely by this code.
- `JsonlEventStore` - portable export/import, and the store used by unit tests
  so kernel tests need no infrastructure.

Both must behave identically. `tests/unit/test_store_conformance.py` runs the
same suite against each, because an export that folds to a different state than
its source would make `ases export` worthless as an audit artifact.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Protocol, runtime_checkable
from uuid import UUID

from ases.kernel.events import ChainProblem, ChainVerification, Event, UnsealedEvent
from ases.kernel.hashing import GENESIS_HASH


class EventStoreError(RuntimeError):
    """Base class for store failures."""


class ChainForkError(EventStoreError):
    """An append was attempted against a stale head.

    Raised when two writers race for the same sequence number. The chain must
    have exactly one writer per run; this is the detection, not the prevention.
    """


class RunNotFoundError(EventStoreError):
    pass


@runtime_checkable
class EventStore(Protocol):
    """Append-only, per-run ordered event log."""

    async def append(self, event: UnsealedEvent) -> Event:
        """Seal and persist one event. Returns the sealed form."""
        ...

    async def append_all(self, events: Sequence[UnsealedEvent]) -> tuple[Event, ...]:
        """Seal and persist several events atomically with respect to the chain.

        Used where a single logical transition produces more than one event and
        an interleaved write would make the log misleading.
        """
        ...

    def read(self, run_id: UUID, *, after_seq: int = 0) -> AsyncIterator[Event]:
        """Stream a run's events in sequence order."""
        ...

    async def read_all(self, run_id: UUID) -> tuple[Event, ...]: ...

    async def head(self, run_id: UUID) -> Event | None:
        """Most recent event, or None for a run with no events yet."""
        ...

    async def verify_chain(self, run_id: UUID) -> ChainVerification:
        """Recompute every digest and link. Tamper-evidence must be checked."""
        ...

    async def list_runs(self) -> tuple[UUID, ...]: ...


def verify_sequence(run_id: UUID, events: Iterable[Event]) -> ChainVerification:
    """Shared verification used by every store implementation.

    Checks four independent properties. They catch different failures, so all
    four are needed:

    1. `seq` is dense and starts at 1  - detects deletion
    2. the first event links to GENESIS - detects a truncated head
    3. each `prev_hash` equals the predecessor's `hash` - detects reordering
    4. each `hash` recomputes from its own fields - detects field edits
    """
    problems: list[ChainProblem] = []
    previous: Event | None = None
    count = 0

    for event in events:
        count += 1
        expected_seq = 1 if previous is None else previous.seq + 1
        if event.seq != expected_seq:
            problems.append(
                ChainProblem(
                    seq=event.seq,
                    reason=f"expected seq {expected_seq}, found {event.seq}",
                )
            )

        expected_prev = GENESIS_HASH if previous is None else previous.hash
        if event.prev_hash != expected_prev:
            problems.append(
                ChainProblem(
                    seq=event.seq,
                    reason=(
                        "broken link: prev_hash "
                        f"{event.prev_hash[:12]}... != {expected_prev[:12]}..."
                    ),
                )
            )

        recomputed = event.recompute_hash()
        if recomputed != event.hash:
            problems.append(
                ChainProblem(
                    seq=event.seq,
                    reason=(
                        "content altered: recomputed "
                        f"{recomputed[:12]}... != stored {event.hash[:12]}..."
                    ),
                )
            )

        if event.run_id != run_id:
            problems.append(
                ChainProblem(seq=event.seq, reason=f"event belongs to run {event.run_id}")
            )

        previous = event

    return ChainVerification(run_id=run_id, events_checked=count, problems=tuple(problems))
