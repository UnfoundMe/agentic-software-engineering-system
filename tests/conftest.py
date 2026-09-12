"""Shared fixtures.

Unit tests must not require infrastructure - that is what lets the kernel be
verified with no database and no LLM in the process.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from ases.kernel.events import Actor, EventType, UnsealedEvent
from ases.kernel.store.jsonl import JsonlEventStore


@pytest.fixture
def run_id() -> UUID:
    return uuid4()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[JsonlEventStore]:
    # fsync is correct for a real audit log and needlessly slow for tests.
    yield JsonlEventStore(tmp_path / "runs", fsync=False)


def make_event(
    run_id: UUID,
    event_type: EventType,
    *,
    node_id: str | None = None,
    attempt: int | None = None,
    **payload: object,
) -> UnsealedEvent:
    return UnsealedEvent(
        run_id=run_id,
        type=event_type,
        actor=Actor.kernel(),
        node_id=node_id,
        attempt=attempt,
        payload=payload,
    )
