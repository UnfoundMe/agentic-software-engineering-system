"""`ases runs list`: surfaces run status/rejection reasons that the raw
sandboxes/ directory (just git worktrees named by run_id) does not show.

Unit tests must not require infrastructure, so `PostgresEventStore` is
substituted with a `JsonlEventStore`-backed fake that speaks the same async
`list_runs`/`read_all`/`close` surface the CLI calls.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from rich.console import Console
from typer.testing import CliRunner

from ases.interfaces import cli
from ases.kernel.events import Actor, Event, EventType, UnsealedEvent
from ases.kernel.state import (
    ApprovalRecord,
    NodeState,
    NodeStatus,
    RunState,
    RunStatus,
)
from ases.kernel.store.jsonl import JsonlEventStore

# ---------------------------------------------------------------------------
# _rejection_reasons: pure function over a folded RunState
# ---------------------------------------------------------------------------


def _state(run_id: UUID, **overrides: object) -> RunState:
    return RunState(run_id=run_id, **overrides)  # type: ignore[arg-type]


def test_rejection_reasons_empty_for_a_clean_run(run_id: UUID) -> None:
    state = _state(run_id, status=RunStatus.COMPLETED)
    assert cli._rejection_reasons(state) == []


def test_rejection_reasons_includes_rejected_approval_with_reason(run_id: UUID) -> None:
    state = _state(run_id, status=RunStatus.RUNNING)
    state.approvals["gate3"] = ApprovalRecord(
        node_id="gate3",
        artifact_hash="abc",
        granted=False,
        actor="human:reviewer",
        decided_at=datetime.now(UTC),
        reason="missing rate limiting",
    )
    reasons = cli._rejection_reasons(state)
    assert reasons == ["gate3 rejected (missing rate limiting)"]


def test_rejection_reasons_includes_revoked_approval(run_id: UUID) -> None:
    state = _state(run_id, status=RunStatus.RUNNING)
    state.approvals["gate3"] = ApprovalRecord(
        node_id="gate3",
        artifact_hash="abc",
        granted=False,
        actor="human:reviewer",
        decided_at=datetime.now(UTC),
        revoked=True,
        revoked_reason="upstream artifact changed",
    )
    reasons = cli._rejection_reasons(state)
    assert reasons == ["gate3 revoked (upstream artifact changed)"]


def test_rejection_reasons_includes_halt_reason(run_id: UUID) -> None:
    state = _state(run_id, status=RunStatus.HALTED, halt_reason="budget exceeded")
    assert cli._rejection_reasons(state) == ["halted: budget exceeded"]


def test_rejection_reasons_includes_failed_node_last_error(run_id: UUID) -> None:
    state = _state(run_id, status=RunStatus.FAILED)
    state.nodes["implement"] = NodeState(
        node_id="implement", status=NodeStatus.FAILED, last_error="dotnet build failed"
    )
    assert cli._rejection_reasons(state) == ["implement failed: dotnet build failed"]


def test_rejection_reasons_combines_all_sources_in_order(run_id: UUID) -> None:
    state = _state(run_id, status=RunStatus.HALTED, halt_reason="budget exceeded")
    state.approvals["gate3"] = ApprovalRecord(
        node_id="gate3",
        artifact_hash="abc",
        granted=False,
        actor="human:reviewer",
        decided_at=datetime.now(UTC),
        reason="not ready",
    )
    state.nodes["implement"] = NodeState(
        node_id="implement", status=NodeStatus.FAILED, last_error="dotnet build failed"
    )
    assert cli._rejection_reasons(state) == [
        "gate3 rejected (not ready)",
        "halted: budget exceeded",
        "implement failed: dotnet build failed",
    ]


def test_rejection_reasons_granted_approval_is_not_a_reason(run_id: UUID) -> None:
    state = _state(run_id, status=RunStatus.COMPLETED)
    state.approvals["gate3"] = ApprovalRecord(
        node_id="gate3",
        artifact_hash="abc",
        granted=True,
        actor="human:reviewer",
        decided_at=datetime.now(UTC),
    )
    assert cli._rejection_reasons(state) == []


# ---------------------------------------------------------------------------
# `ases runs list`: end-to-end against a fake store
# ---------------------------------------------------------------------------


class _FakeEventStore:
    """Duck-types `PostgresEventStore`'s async surface over a JsonlEventStore
    so `_runs_list` can be exercised with no database."""

    def __init__(self, root: Path) -> None:
        self._inner = JsonlEventStore(root, fsync=False)

    async def list_runs(self) -> tuple[UUID, ...]:
        return await self._inner.list_runs()

    async def read_all(self, run_id: UUID) -> tuple[Event, ...]:
        return await self._inner.read_all(run_id)

    async def close(self) -> None:
        return None


async def _seed(root: Path, run_id: UUID, *specs: tuple[EventType, dict[str, object]]) -> None:
    store = JsonlEventStore(root, fsync=False)
    events: list[UnsealedEvent] = []
    for event_type, kwargs in specs:
        node_id = kwargs.pop("node_id", None)
        events.append(
            UnsealedEvent(
                run_id=run_id,
                type=event_type,
                actor=Actor.kernel(),
                node_id=node_id,  # type: ignore[arg-type]
                payload=kwargs,
            )
        )
    await store.append_all(events)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_runs_list_shows_status_sandbox_and_rejection_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: CliRunner, run_id: UUID
) -> None:
    events_root = tmp_path / "events"
    sandboxes_root = tmp_path / "sandboxes"
    sandboxes_root.mkdir()
    (sandboxes_root / str(run_id)).mkdir()  # sandbox still on disk

    asyncio.run(
        _seed(
            events_root,
            run_id,
            (EventType.RUN_CREATED, {"workflow": "greenfield"}),
            (EventType.RUN_STARTED, {}),
            (EventType.NODE_READY, {"node_id": "gate3"}),
            (EventType.NODE_STARTED, {"node_id": "gate3", "attempt": 1}),
            (EventType.APPROVAL_REQUESTED, {"node_id": "gate3", "artifact_hash": "abc"}),
            (
                EventType.APPROVAL_REJECTED,
                {"node_id": "gate3", "artifact_hash": "abc", "reason": "missing rate limiting"},
            ),
            (EventType.RUN_HALTED, {"reason": "manual stop"}),
        )
    )

    monkeypatch.setattr(cli, "PostgresEventStore", lambda _dsn: _FakeEventStore(events_root))
    monkeypatch.setattr(cli, "SANDBOXES_DIR", sandboxes_root)
    monkeypatch.setattr(cli, "console", Console(width=400))

    result = runner.invoke(cli.app, ["runs", "list"])

    assert result.exit_code == 0, result.output
    assert str(run_id) in result.output
    assert "halted" in result.output
    assert str(sandboxes_root / str(run_id)) in result.output
    assert "gate3 rejected (missing rate limiting)" in result.output
    assert "halted: manual stop" in result.output


def test_runs_list_marks_missing_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: CliRunner, run_id: UUID
) -> None:
    events_root = tmp_path / "events"
    sandboxes_root = tmp_path / "sandboxes"  # created, but no run_id subdir

    asyncio.run(
        _seed(
            events_root,
            run_id,
            (EventType.RUN_CREATED, {"workflow": "greenfield"}),
            (EventType.RUN_STARTED, {}),
            (EventType.RUN_COMPLETED, {}),
        )
    )

    monkeypatch.setattr(cli, "PostgresEventStore", lambda _dsn: _FakeEventStore(events_root))
    monkeypatch.setattr(cli, "SANDBOXES_DIR", sandboxes_root)
    monkeypatch.setattr(cli, "console", Console(width=400))

    result = runner.invoke(cli.app, ["runs", "list"])

    assert result.exit_code == 0, result.output
    assert str(run_id) in result.output
    assert "completed" in result.output
    assert "removed/none" in result.output


def test_runs_list_marks_a_malformed_run_unreadable_without_losing_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: CliRunner, run_id: UUID
) -> None:
    """Found live: `PostgresEventStore` held a run whose event log did not
    fold (an approval decision recorded with no preceding node lifecycle
    events - likely a leftover from a test/tool writing directly to the same
    physical database `settings().app_dsn` points at, not a real orchestrator
    run) - the old implementation folded every run in one list comprehension,
    so that one run crashed the listing for every run, including healthy
    ones. One bad run must now show as `unreadable` and nothing else."""
    events_root = tmp_path / "events"
    sandboxes_root = tmp_path / "sandboxes"
    healthy_run_id = uuid4()

    # Malformed: an approval decision with no NODE_READY/NODE_STARTED first -
    # `kernel.state.apply` cannot legally transition a PENDING node straight
    # to SUCCEEDED, which is exactly the shape this reproduces.
    asyncio.run(
        _seed(
            events_root,
            run_id,
            (EventType.RUN_CREATED, {"workflow": "greenfield"}),
            (
                EventType.APPROVAL_GRANTED,
                {"node_id": "gate1", "artifact_hash": "abc", "reason": ""},
            ),
        )
    )
    asyncio.run(
        _seed(
            events_root,
            healthy_run_id,
            (EventType.RUN_CREATED, {"workflow": "greenfield"}),
            (EventType.RUN_STARTED, {}),
            (EventType.RUN_COMPLETED, {}),
        )
    )

    monkeypatch.setattr(cli, "PostgresEventStore", lambda _dsn: _FakeEventStore(events_root))
    monkeypatch.setattr(cli, "SANDBOXES_DIR", sandboxes_root)
    monkeypatch.setattr(cli, "console", Console(width=400))

    result = runner.invoke(cli.app, ["runs", "list"])

    assert result.exit_code == 0, result.output
    assert str(run_id) in result.output
    assert "unreadable" in result.output
    assert "InvalidTransitionError" in result.output
    # The healthy run is unaffected by the other one's malformed log.
    assert str(healthy_run_id) in result.output
    assert "completed" in result.output


def test_runs_list_reports_no_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    events_root = tmp_path / "events"
    monkeypatch.setattr(cli, "PostgresEventStore", lambda _dsn: _FakeEventStore(events_root))
    monkeypatch.setattr(cli, "SANDBOXES_DIR", tmp_path / "sandboxes")

    result = runner.invoke(cli.app, ["runs", "list"])

    assert result.exit_code == 0, result.output
    assert "no runs recorded" in result.output
