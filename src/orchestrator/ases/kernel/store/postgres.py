"""The Postgres-backed event store: the source of truth (docs/04).

Connects as `ases_app` - DML only, no DDL, and (empirically verified against
the live schema in `tests/integration/test_store_postgres.py`) no UPDATE or
DELETE on `control.events` even if attempted: the grant was never given, and
the append-only trigger blocks it a second time even for the table's owner.
This module cannot violate that boundary by construction, not merely by
convention - there is no code path here that could even ask for UPDATE/DELETE
to succeed, since the role behind every connection this module opens lacks
the privilege outright.

Concurrency: within one process, multiple coroutines appending to the same
`run_id` are serialized by a Postgres session-level advisory lock
(`pg_advisory_xact_lock`), held for the transaction and released automatically
on commit or rollback. This is the one property `JsonlEventStore`'s in-process
`asyncio.Lock` cannot provide and this store must: two separate *processes*
(two orchestrator workers, or a crashed-and-restarted one racing its own
successor) appending to the same run must not fork the chain either.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ases.kernel.events import Actor, ChainVerification, Event, UnsealedEvent
from ases.kernel.hashing import GENESIS_HASH
from ases.kernel.store.base import ChainForkError, EventStoreError, verify_sequence


def _advisory_lock_key(run_id: UUID) -> int:
    """A stable 64-bit signed key for `pg_advisory_xact_lock`, derived from
    the run_id. Collisions are possible in principle (two different run_ids
    hashing to the same key) and merely cost unrelated runs a lock they did
    not need - never a correctness problem, since the lock only ever gates
    entry to a critical section that also re-checks `run_id` explicitly."""
    return int.from_bytes(run_id.bytes[:8], "big", signed=True)


def _row_to_event(row: object) -> Event:
    mapping = row._mapping  # type: ignore[attr-defined]
    return Event(
        seq=mapping["seq"],
        event_id=mapping["event_id"],
        run_id=mapping["run_id"],
        type=mapping["type"],
        actor=Actor.model_validate(_maybe_loads(mapping["actor"])),
        payload=_maybe_loads(mapping["payload"]),
        node_id=mapping["node_id"],
        attempt=mapping["attempt"],
        created_at=mapping["created_at"],
        prev_hash=mapping["prev_hash"],
        hash=mapping["hash"],
    )


def _maybe_loads(value: object) -> dict[str, object]:
    """asyncpg/SQLAlchemy return JSONB as an already-decoded dict in most
    configurations, but as a raw string in others depending on driver
    version - handling both keeps this store correct either way."""
    if isinstance(value, str):
        loaded: dict[str, object] = json.loads(value)
        return loaded
    assert isinstance(value, dict)
    return value


class PostgresEventStore:
    """`EventStore` backed by `control.events`. See the module docstring for
    the concurrency model and the security properties this relies on."""

    def __init__(self, dsn: str) -> None:
        self._engine: AsyncEngine = create_async_engine(dsn, pool_pre_ping=True)

    async def close(self) -> None:
        await self._engine.dispose()

    # -- writing ---------------------------------------------------------

    async def append(self, event: UnsealedEvent) -> Event:
        sealed = await self.append_all([event])
        return sealed[0]

    async def append_all(self, events: Sequence[UnsealedEvent]) -> tuple[Event, ...]:
        if not events:
            return ()
        run_ids = {e.run_id for e in events}
        if len(run_ids) != 1:
            raise EventStoreError("append_all requires all events to share one run_id")
        run_id = run_ids.pop()

        async with self._engine.begin() as conn:
            # Held for the transaction; released automatically on commit or
            # rollback. This is what makes two processes appending to the
            # same run_id safe - JsonlEventStore's in-process asyncio.Lock
            # cannot reach across a process boundary the way this can.
            await conn.execute(
                text("SELECT pg_advisory_xact_lock(:key)"), {"key": _advisory_lock_key(run_id)}
            )

            head_row = (
                await conn.execute(
                    text(
                        "SELECT seq, hash FROM control.events "
                        "WHERE run_id = :run_id ORDER BY seq DESC LIMIT 1"
                    ),
                    {"run_id": run_id},
                )
            ).first()
            seq = head_row.seq if head_row else 0
            prev_hash = head_row.hash if head_row else GENESIS_HASH

            sealed: list[Event] = []
            for unsealed in events:
                seq += 1
                item = Event.seal(unsealed, seq=seq, prev_hash=prev_hash)
                prev_hash = item.hash
                sealed.append(item)

            for item in sealed:
                try:
                    await conn.execute(
                        text("""
                            INSERT INTO control.events
                                (run_id, seq, event_id, type, actor, payload,
                                 node_id, attempt, created_at, prev_hash, hash)
                            VALUES
                                (:run_id, :seq, :event_id, :type, :actor, :payload,
                                 :node_id, :attempt, :created_at, :prev_hash, :hash)
                        """),
                        {
                            "run_id": item.run_id,
                            "seq": item.seq,
                            "event_id": item.event_id,
                            "type": item.type.value,
                            "actor": json.dumps(
                                {"kind": item.actor.kind.value, "id": item.actor.id}
                            ),
                            "payload": json.dumps(dict(item.payload)),
                            "node_id": item.node_id,
                            "attempt": item.attempt,
                            "created_at": item.created_at,
                            "prev_hash": item.prev_hash,
                            "hash": item.hash,
                        },
                    )
                except Exception as exc:
                    # A unique-constraint violation on (run_id, seq) or
                    # (run_id, prev_hash) means another writer beat this one
                    # to the same position - the fork the advisory lock exists
                    # to prevent, surfaced anyway as defence in depth (e.g. a
                    # second store instance not going through this class).
                    raise ChainForkError(
                        f"append to run {run_id} at seq {item.seq} conflicted: {exc}"
                    ) from exc

        return tuple(sealed)

    # -- reading -----------------------------------------------------------

    async def read(self, run_id: UUID, *, after_seq: int = 0) -> AsyncIterator[Event]:
        for event in await self.read_all(run_id):
            if event.seq > after_seq:
                yield event

    async def read_all(self, run_id: UUID) -> tuple[Event, ...]:
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT seq, event_id, run_id, type, actor, payload, node_id, "
                        "attempt, created_at, prev_hash, hash FROM control.events "
                        "WHERE run_id = :run_id ORDER BY seq ASC"
                    ),
                    {"run_id": run_id},
                )
            ).all()
        return tuple(_row_to_event(row) for row in rows)

    async def head(self, run_id: UUID) -> Event | None:
        events = await self.read_all(run_id)
        return events[-1] if events else None

    # -- integrity -----------------------------------------------------

    async def verify_chain(self, run_id: UUID) -> ChainVerification:
        return verify_sequence(run_id, await self.read_all(run_id))

    async def list_runs(self) -> tuple[UUID, ...]:
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text("SELECT DISTINCT run_id FROM control.events"))).all()
        return tuple(row.run_id for row in rows)
