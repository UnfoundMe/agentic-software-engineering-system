"""JSONL event store behaviour and chain verification."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from ases.kernel.events import Event, EventType
from ases.kernel.hashing import GENESIS_HASH
from ases.kernel.store.base import EventStore, EventStoreError
from ases.kernel.store.jsonl import JsonlEventStore
from tests.conftest import make_event


async def test_store_satisfies_the_protocol(store: JsonlEventStore) -> None:
    assert isinstance(store, EventStore)


async def test_append_assigns_dense_sequence_and_links(
    store: JsonlEventStore, run_id: UUID
) -> None:
    first = await store.append(make_event(run_id, EventType.RUN_CREATED))
    second = await store.append(make_event(run_id, EventType.RUN_STARTED))

    assert (first.seq, second.seq) == (1, 2)
    assert first.prev_hash == GENESIS_HASH
    assert second.prev_hash == first.hash


async def test_verify_chain_passes_on_an_untouched_log(
    store: JsonlEventStore, run_id: UUID
) -> None:
    for event_type in (EventType.RUN_CREATED, EventType.RUN_STARTED, EventType.RUN_COMPLETED):
        await store.append(make_event(run_id, event_type))

    result = await store.verify_chain(run_id)
    assert result.ok
    assert result.events_checked == 3


async def test_verify_chain_detects_an_edited_payload(
    store: JsonlEventStore, run_id: UUID, tmp_path: Path
) -> None:
    """The scenario the hash chain exists for: someone edits the audit log."""
    await store.append(make_event(run_id, EventType.RUN_CREATED, workflow="greenfield"))
    await store.append(make_event(run_id, EventType.RUN_STARTED))

    path = tmp_path / "runs" / str(run_id) / "events.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["payload"]["workflow"] = "something_else"
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = await store.verify_chain(run_id)
    assert not result.ok
    assert any("content altered" in p.reason for p in result.problems)


async def test_verify_chain_detects_a_deleted_event(
    store: JsonlEventStore, run_id: UUID, tmp_path: Path
) -> None:
    for event_type in (EventType.RUN_CREATED, EventType.RUN_STARTED, EventType.RUN_COMPLETED):
        await store.append(make_event(run_id, event_type))

    path = tmp_path / "runs" / str(run_id) / "events.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    del lines[1]  # remove the middle event
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = await store.verify_chain(run_id)
    assert not result.ok
    reasons = " ".join(p.reason for p in result.problems)
    assert "expected seq" in reasons or "broken link" in reasons


async def test_corrupt_line_is_surfaced_not_skipped(
    store: JsonlEventStore, run_id: UUID, tmp_path: Path
) -> None:
    """Silently skipping a bad line would make a damaged log look healthy."""
    await store.append(make_event(run_id, EventType.RUN_CREATED))
    path = tmp_path / "runs" / str(run_id) / "events.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not json}\n")

    with pytest.raises(EventStoreError, match="not a valid event"):
        await store.read_all(run_id)


async def test_append_all_is_contiguous(store: JsonlEventStore, run_id: UUID) -> None:
    events = await store.append_all(
        [
            make_event(run_id, EventType.NODE_READY, node_id="a"),
            make_event(run_id, EventType.NODE_STARTED, node_id="a"),
        ]
    )
    assert [e.seq for e in events] == [1, 2]
    assert events[1].prev_hash == events[0].hash


async def test_append_all_rejects_mixed_runs(store: JsonlEventStore, run_id: UUID) -> None:
    with pytest.raises(EventStoreError, match="one run_id"):
        await store.append_all(
            [
                make_event(run_id, EventType.RUN_CREATED),
                make_event(uuid4(), EventType.RUN_CREATED),
            ]
        )


async def test_concurrent_appends_do_not_fork_the_chain(
    store: JsonlEventStore, run_id: UUID
) -> None:
    """The per-run lock is what keeps `prev_hash` trivially correct.

    Without it two tasks would read the same head and both claim the same seq,
    producing a fork that verification would reject.
    """
    await asyncio.gather(
        *(
            store.append(make_event(run_id, EventType.NODE_READY, node_id=f"n{i}"))
            for i in range(25)
        )
    )
    result = await store.verify_chain(run_id)
    assert result.ok, [p.reason for p in result.problems]
    assert result.events_checked == 25


async def test_read_after_seq_streams_the_tail(store: JsonlEventStore, run_id: UUID) -> None:
    for _ in range(5):
        await store.append(make_event(run_id, EventType.NODE_READY, node_id="a"))
    tail = [e.seq async for e in store.read(run_id, after_seq=3)]
    assert tail == [4, 5]


async def test_unknown_run_reads_empty(store: JsonlEventStore) -> None:
    assert await store.read_all(uuid4()) == ()
    assert await store.head(uuid4()) is None


async def test_list_runs_ignores_non_run_directories(
    store: JsonlEventStore, run_id: UUID, tmp_path: Path
) -> None:
    await store.append(make_event(run_id, EventType.RUN_CREATED))
    (tmp_path / "runs" / "not-a-uuid").mkdir(parents=True, exist_ok=True)
    assert await store.list_runs() == (run_id,)


async def test_round_trip_through_disk_preserves_the_event(
    store: JsonlEventStore, run_id: UUID
) -> None:
    """Export is only useful as an audit artifact if it reloads identically."""
    original = await store.append(
        make_event(run_id, EventType.ARTIFACT_PRODUCED, node_id="req", artifact_hash="abc")
    )
    (reloaded,) = await store.read_all(run_id)
    assert reloaded == original
    assert isinstance(reloaded, Event)
    assert reloaded.recompute_hash() == reloaded.hash
