"""File-backed event store: one JSONL file per run.

Purpose is twofold:

- **Portability.** `ases export` writes this format so a run's audit trail can
  be archived, shipped or reviewed without a database. `ases replay` folds it
  back with no Postgres involved, which is the practical demonstration that
  state really is a fold and not a separately maintained structure.
- **Testability.** Kernel unit tests use it so they require no infrastructure.

Single-process only. There is no cross-process locking, so this store must not
back a live orchestrator run; `PostgresEventStore` is the source of truth.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from uuid import UUID

from ases.kernel.events import ChainVerification, Event, UnsealedEvent
from ases.kernel.hashing import GENESIS_HASH
from ases.kernel.store.base import EventStoreError, verify_sequence

_FILENAME = "events.jsonl"


class JsonlEventStore:
    """Append-only JSONL store rooted at a directory of run folders."""

    def __init__(self, root: Path, *, fsync: bool = True) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        # One lock per run. The chain has exactly one writer by construction;
        # the lock makes that true for concurrent tasks in this process.
        self._locks: dict[UUID, asyncio.Lock] = {}
        # fsync on every append is deliberate for an audit log - a lost tail
        # after a crash would silently truncate the chain. Tests disable it.
        self._fsync = fsync

    # -- paths ---------------------------------------------------------

    def _path(self, run_id: UUID) -> Path:
        return self._root / str(run_id) / _FILENAME

    def _lock(self, run_id: UUID) -> asyncio.Lock:
        return self._locks.setdefault(run_id, asyncio.Lock())

    # -- writing -------------------------------------------------------

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

        async with self._lock(run_id):
            head = await self._head_unlocked(run_id)
            seq = head.seq if head else 0
            prev_hash = head.hash if head else GENESIS_HASH

            sealed: list[Event] = []
            for unsealed in events:
                seq += 1
                item = Event.seal(unsealed, seq=seq, prev_hash=prev_hash)
                prev_hash = item.hash
                sealed.append(item)

            await asyncio.to_thread(self._write_lines, run_id, sealed)
            return tuple(sealed)

    def _write_lines(self, run_id: UUID, events: Sequence[Event]) -> None:
        path = self._path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Buffer the whole batch, then one write: a partial batch would leave a
        # chain that verifies as truncated rather than one that is merely short.
        blob = "".join(e.model_dump_json() + "\n" for e in events)
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(blob)
            fh.flush()
            if self._fsync:
                os.fsync(fh.fileno())

    # -- reading -------------------------------------------------------

    async def read(self, run_id: UUID, *, after_seq: int = 0) -> AsyncIterator[Event]:
        for event in await asyncio.to_thread(self._read_sync, run_id):
            if event.seq > after_seq:
                yield event

    async def read_all(self, run_id: UUID) -> tuple[Event, ...]:
        return tuple(await asyncio.to_thread(self._read_sync, run_id))

    def _read_sync(self, run_id: UUID) -> list[Event]:
        path = self._path(run_id)
        if not path.exists():
            return []
        events: list[Event] = []
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(Event.model_validate(json.loads(line)))
                except (json.JSONDecodeError, ValueError) as exc:
                    # Surfaced rather than skipped: a corrupt line in an audit
                    # log is a finding, not noise to be tolerated.
                    raise EventStoreError(f"{path}:{lineno} is not a valid event: {exc}") from exc
        return events

    async def head(self, run_id: UUID) -> Event | None:
        async with self._lock(run_id):
            return await self._head_unlocked(run_id)

    async def _head_unlocked(self, run_id: UUID) -> Event | None:
        events = await asyncio.to_thread(self._read_sync, run_id)
        return events[-1] if events else None

    # -- integrity -----------------------------------------------------

    async def verify_chain(self, run_id: UUID) -> ChainVerification:
        return verify_sequence(run_id, await self.read_all(run_id))

    async def list_runs(self) -> tuple[UUID, ...]:
        found: list[UUID] = []
        for child in sorted(self._root.iterdir()):
            if not (child / _FILENAME).exists():
                continue
            try:
                found.append(UUID(child.name))
            except ValueError:
                continue  # not a run directory
        return tuple(found)
