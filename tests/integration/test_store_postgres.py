"""PostgresEventStore, exercised against the live container.

Runs the same conformance scenarios as `tests/unit/test_store_jsonl.py`
wherever they apply, plus two properties only a real database can prove:
genuine cross-connection concurrency safety, and the security boundaries
docs/04 promises (`ases_app` cannot mutate `control.events`; `workload_app`
cannot see `control` at all).
"""

from __future__ import annotations

import asyncio
from uuid import UUID

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine

from ases.config import settings
from ases.kernel.events import Actor, EventType
from ases.kernel.hashing import GENESIS_HASH
from ases.kernel.store.postgres import PostgresEventStore
from tests.conftest import make_event

pytestmark = pytest.mark.integration


async def test_append_assigns_dense_sequence_and_links(
    pg_store: PostgresEventStore, run_id: UUID
) -> None:
    first = await pg_store.append(make_event(run_id, EventType.RUN_CREATED, workflow="greenfield"))
    second = await pg_store.append(make_event(run_id, EventType.RUN_STARTED))

    assert (first.seq, second.seq) == (1, 2)
    assert first.prev_hash == GENESIS_HASH
    assert second.prev_hash == first.hash


async def test_verify_chain_passes_on_an_untouched_log(
    pg_store: PostgresEventStore, run_id: UUID
) -> None:
    for event_type in (EventType.RUN_CREATED, EventType.RUN_STARTED, EventType.RUN_COMPLETED):
        await pg_store.append(make_event(run_id, event_type))

    result = await pg_store.verify_chain(run_id)
    assert result.ok
    assert result.events_checked == 3


async def test_round_trip_preserves_actor_and_payload(
    pg_store: PostgresEventStore, run_id: UUID
) -> None:
    original = await pg_store.append(
        make_event(
            run_id,
            EventType.APPROVAL_GRANTED,
            node_id="gate1",
            actor=Actor.human("alice"),
            artifact_hash="abc123",
            reason="looks correct",
        )
    )
    (reloaded,) = await pg_store.read_all(run_id)
    assert reloaded == original
    assert reloaded.actor == Actor.human("alice")
    assert reloaded.payload["reason"] == "looks correct"
    assert reloaded.recompute_hash() == reloaded.hash


async def test_read_after_seq_streams_the_tail(pg_store: PostgresEventStore, run_id: UUID) -> None:
    for _ in range(5):
        await pg_store.append(make_event(run_id, EventType.NODE_READY, node_id="a"))
    tail = [e.seq async for e in pg_store.read(run_id, after_seq=3)]
    assert tail == [4, 5]


async def test_unknown_run_reads_empty(pg_store: PostgresEventStore) -> None:
    from uuid import uuid4

    assert await pg_store.read_all(uuid4()) == ()
    assert await pg_store.head(uuid4()) is None


async def test_concurrent_appends_across_real_connections_do_not_fork_the_chain(
    run_id: UUID,
) -> None:
    """The property JsonlEventStore's in-process asyncio.Lock cannot prove:
    this uses N *separate* PostgresEventStore instances, each with its own
    connection, appending to the same run_id at once. Only the database-side
    advisory lock can serialize this correctly."""
    stores = [PostgresEventStore(settings().app_dsn) for _ in range(10)]
    try:
        await asyncio.gather(
            *(
                store.append(make_event(run_id, EventType.NODE_READY, node_id=f"n{i}"))
                for i, store in enumerate(stores)
            )
        )
    finally:
        for store in stores:
            await store.close()

    verifier = PostgresEventStore(settings().app_dsn)
    try:
        result = await verifier.verify_chain(run_id)
        assert result.ok, [p.reason for p in result.problems]
        assert result.events_checked == 10
    finally:
        await verifier.close()


async def test_defence_in_depth_tamper_is_still_detected(
    pg_store: PostgresEventStore, run_id: UUID
) -> None:
    """The append-only trigger and the missing UPDATE/DELETE grant are the
    primary defence (proven directly by the two security tests below). This
    proves the *secondary* one: even if a superuser mistake bypassed both -
    the only way to edit a row at all, confirmed empirically - verify_chain
    still catches it. Defence in depth is only real if the second layer
    actually works when the first is disabled, not merely present."""
    await pg_store.append(make_event(run_id, EventType.RUN_CREATED, workflow="greenfield"))
    await pg_store.append(
        make_event(run_id, EventType.APPROVAL_REJECTED, node_id="gate1", reason="too vague")
    )

    su_engine = create_async_engine(settings().superuser_dsn)
    try:
        async with su_engine.begin() as conn:
            await conn.execute(
                text("ALTER TABLE control.events DISABLE TRIGGER events_append_only")
            )
            await conn.execute(
                text(
                    'UPDATE control.events SET payload = payload || \'{"reason": "looked fine"}\' '
                    "WHERE run_id = :run_id AND type = 'approval.rejected'"
                ),
                {"run_id": run_id},
            )
            await conn.execute(text("ALTER TABLE control.events ENABLE TRIGGER events_append_only"))
    finally:
        await su_engine.dispose()

    result = await pg_store.verify_chain(run_id)
    assert not result.ok
    assert any("content altered" in p.reason for p in result.problems)


async def test_ases_app_cannot_update_or_delete_events() -> None:
    """The grant-level half of append-only enforcement, direct from docs/04
    section 3.2 - not exercised through PostgresEventStore's own API (which
    never issues UPDATE/DELETE), but against the same role it connects as."""
    engine = create_async_engine(settings().app_dsn)
    try:
        async with engine.connect() as conn:
            with pytest.raises(DBAPIError, match="permission denied"):
                await conn.execute(text("UPDATE control.events SET hash = 'x'"))
        async with engine.connect() as conn:
            with pytest.raises(DBAPIError, match="permission denied"):
                await conn.execute(text("DELETE FROM control.events"))
    finally:
        await engine.dispose()


async def test_workload_app_cannot_see_the_control_schema() -> None:
    """The isolation the entire database design rests on (docs/04 section
    1.1): the workload role has no USAGE on `control` at all."""
    engine = create_async_engine(settings().workload_app_dsn)
    try:
        async with engine.connect() as conn:
            with pytest.raises(DBAPIError, match="permission denied for schema control"):
                await conn.execute(text("SELECT * FROM control.events"))
    finally:
        await engine.dispose()
