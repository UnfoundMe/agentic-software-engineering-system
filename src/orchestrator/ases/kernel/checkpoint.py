"""Checkpoints: a disposable cache over the fold, never the source of truth.

`RunState = fold(events)` is correct and total on its own; a checkpoint exists
only so that resuming a long run does not mean re-folding its entire history
from event 1. Concretely: `resume(store, run_id)` with no checkpoint present
folds everything; with a checkpoint present it folds only the events after
`checkpoint.seq` and applies them on top of the saved state. Both paths must
produce the same `RunState` - that equivalence is the property this module
exists to provide, and it is asserted directly in
`tests/unit/test_checkpoint.py`.

Deleting every checkpoint file must never lose information, only re-fold time.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from ases.kernel.state import RunState, apply
from ases.kernel.store.base import EventStore


class Checkpoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: UUID
    seq: int
    state: RunState


class CheckpointStore:
    """File-backed checkpoint cache, one JSON file per run.

    Not a `Protocol` (unlike `EventStore`): there is exactly one reasonable
    implementation shape here, and no second backend is anticipated the way
    Postgres eventually replaces `JsonlEventStore` for events.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: UUID) -> Path:
        return self._root / str(run_id) / "checkpoint.json"

    async def save(self, checkpoint: Checkpoint) -> None:
        await asyncio.to_thread(self._save_sync, checkpoint)

    def _save_sync(self, checkpoint: Checkpoint) -> None:
        path = self._path(checkpoint.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file and replace atomically: a checkpoint that is
        # half-written on disk must never be mistaken for a valid one.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(checkpoint.model_dump_json(), encoding="utf-8")
        tmp.replace(path)

    async def load(self, run_id: UUID) -> Checkpoint | None:
        return await asyncio.to_thread(self._load_sync, run_id)

    def _load_sync(self, run_id: UUID) -> Checkpoint | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        try:
            return Checkpoint.model_validate_json(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            # A corrupt checkpoint is not corruption of the audit log - the
            # log is untouched - so this degrades to "no checkpoint" rather
            # than raising. Re-folding from scratch is always correct.
            return None

    async def delete(self, run_id: UUID) -> None:
        await asyncio.to_thread(self._path(run_id).unlink, True)


async def resume(
    store: EventStore,
    run_id: UUID,
    *,
    checkpoints: CheckpointStore | None = None,
) -> RunState:
    """Reconstruct run state, using a checkpoint as a shortcut when available.

    Equivalent to `fold(run_id, await store.read_all(run_id))` in every case;
    the checkpoint only changes how much of the log is replayed.
    """
    checkpoint = await checkpoints.load(run_id) if checkpoints else None
    if checkpoint is None:
        events = await store.read_all(run_id)
        state = RunState(run_id=run_id)
        for event in events:
            apply(state, event)
        return state

    state = checkpoint.state.model_copy(deep=True)
    async for event in store.read(run_id, after_seq=checkpoint.seq):
        apply(state, event)
    return state
