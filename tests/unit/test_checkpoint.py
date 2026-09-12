"""Checkpoints: disposable, and equivalent to a full re-fold when present."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from ases.kernel.checkpoint import Checkpoint, CheckpointStore, resume
from ases.kernel.events import EventType
from ases.kernel.state import fold
from ases.kernel.store.jsonl import JsonlEventStore
from tests.conftest import make_event


async def test_resume_with_no_checkpoint_equals_full_fold(
    store: JsonlEventStore, run_id: UUID
) -> None:
    for event_type in (EventType.RUN_CREATED, EventType.RUN_STARTED):
        await store.append(make_event(run_id, event_type))

    via_resume = await resume(store, run_id, checkpoints=None)
    via_fold = fold(run_id, await store.read_all(run_id))
    assert via_resume.model_dump() == via_fold.model_dump()


async def test_resume_with_a_checkpoint_equals_full_fold(
    store: JsonlEventStore, run_id: UUID, tmp_path: Path
) -> None:
    checkpoints = CheckpointStore(tmp_path / "cp")
    await store.append(make_event(run_id, EventType.RUN_CREATED, workflow="greenfield"))
    await store.append(make_event(run_id, EventType.RUN_STARTED))

    halfway = fold(run_id, await store.read_all(run_id))
    await checkpoints.save(Checkpoint(run_id=run_id, seq=halfway.last_seq, state=halfway))

    # More events land after the checkpoint was taken.
    await store.append(make_event(run_id, EventType.NODE_READY, node_id="a"))
    await store.append(make_event(run_id, EventType.NODE_STARTED, node_id="a"))

    via_resume = await resume(store, run_id, checkpoints=checkpoints)
    via_fold = fold(run_id, await store.read_all(run_id))
    assert via_resume.model_dump() == via_fold.model_dump()


async def test_deleting_the_checkpoint_changes_nothing_but_replay_cost(
    store: JsonlEventStore, run_id: UUID, tmp_path: Path
) -> None:
    """The property that justifies calling checkpoints 'disposable'."""
    checkpoints = CheckpointStore(tmp_path / "cp")
    await store.append(make_event(run_id, EventType.RUN_CREATED))
    await store.append(make_event(run_id, EventType.NODE_READY, node_id="a"))

    state = fold(run_id, await store.read_all(run_id))
    await checkpoints.save(Checkpoint(run_id=run_id, seq=state.last_seq, state=state))

    with_checkpoint = await resume(store, run_id, checkpoints=checkpoints)

    await checkpoints.delete(run_id)
    assert await checkpoints.load(run_id) is None

    without_checkpoint = await resume(store, run_id, checkpoints=checkpoints)
    assert with_checkpoint.model_dump() == without_checkpoint.model_dump()


async def test_corrupt_checkpoint_falls_back_to_full_fold(
    store: JsonlEventStore, run_id: UUID, tmp_path: Path
) -> None:
    """A damaged checkpoint file must degrade to 'no checkpoint', never crash
    resume - the event log is untouched and remains authoritative."""
    checkpoints = CheckpointStore(tmp_path / "cp")
    await store.append(make_event(run_id, EventType.RUN_CREATED))

    path = tmp_path / "cp" / str(run_id) / "checkpoint.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")

    state = await resume(store, run_id, checkpoints=checkpoints)
    assert state.status.value == "created"


async def test_missing_checkpoint_for_unknown_run_returns_none(tmp_path: Path) -> None:
    checkpoints = CheckpointStore(tmp_path / "cp")
    from uuid import uuid4

    assert await checkpoints.load(uuid4()) is None
