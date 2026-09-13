"""`interfaces.cli._ProgressExecutor` / `_heartbeat` - found missing live:
between one gate's approval and the next prompt, the terminal printed
nothing at all while an agent was genuinely working (an LLM call, a
`dotnet build`) in the background, indistinguishable from having hung.

Both are exercised directly (not through `CliRunner`) since they are plain
async helpers with no Typer/argument-parsing surface of their own.
"""

from __future__ import annotations

import asyncio
import time
from uuid import UUID

import pytest
from rich.console import Console

from ases.interfaces import cli
from ases.kernel.graph import NodeKind, NodeSpec
from ases.kernel.scheduler import NodeExecutionOutcome
from ases.kernel.state import RunState
from tests.unit.fakes import fail, ok


def _node(node_id: str = "arch") -> NodeSpec:
    return NodeSpec(id=node_id, kind=NodeKind.AGENT, handler="architect", produces="DesignSpec")


class _FixedInner:
    def __init__(self, outcome: NodeExecutionOutcome) -> None:
        self.outcome = outcome
        self.calls = 0

    async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome:
        self.calls += 1
        return self.outcome


async def test_execute_prints_success_with_elapsed_time_and_clears_in_flight(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], run_id: UUID
) -> None:
    monkeypatch.setattr(cli, "console", Console(width=200))
    in_flight: dict[str, float] = {}
    inner = _FixedInner(ok(kind="DesignSpec"))
    executor = cli._ProgressExecutor("architect", inner, in_flight)

    outcome = await executor.execute(_node(), RunState(run_id=run_id))

    assert outcome.ok is True
    assert inner.calls == 1
    assert "arch" not in in_flight
    out = capsys.readouterr().out
    assert "arch" in out
    assert "started" in out
    assert "succeeded" in out


async def test_execute_prints_failure_with_the_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], run_id: UUID
) -> None:
    monkeypatch.setattr(cli, "console", Console(width=200))
    in_flight: dict[str, float] = {}
    executor = cli._ProgressExecutor("architect", _FixedInner(fail("boom")), in_flight)

    outcome = await executor.execute(_node(), RunState(run_id=run_id))

    assert outcome.ok is False
    assert "arch" not in in_flight
    out = capsys.readouterr().out
    assert "failed" in out
    assert "boom" in out


async def test_execute_tracks_the_node_as_in_flight_while_the_inner_call_runs(
    monkeypatch: pytest.MonkeyPatch, run_id: UUID
) -> None:
    monkeypatch.setattr(cli, "console", Console(width=200))
    in_flight: dict[str, float] = {}
    started = asyncio.Event()
    release = asyncio.Event()

    class _SlowInner:
        async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome:
            started.set()
            await release.wait()
            return ok(kind="DesignSpec")

    executor = cli._ProgressExecutor("architect", _SlowInner(), in_flight)
    task = asyncio.create_task(executor.execute(_node(), RunState(run_id=run_id)))

    await started.wait()
    assert "arch" in in_flight

    release.set()
    await task
    assert "arch" not in in_flight


async def test_heartbeat_prints_nothing_when_nothing_is_in_flight(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "console", Console(width=200))
    monkeypatch.setattr(cli, "_HEARTBEAT_SECONDS", 0.01)
    task = asyncio.create_task(cli._heartbeat({}))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert capsys.readouterr().out == ""


async def test_heartbeat_reports_in_flight_nodes_and_elapsed_seconds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "console", Console(width=200))
    monkeypatch.setattr(cli, "_HEARTBEAT_SECONDS", 0.01)
    in_flight = {"arch": time.monotonic()}
    task = asyncio.create_task(cli._heartbeat(in_flight))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    out = capsys.readouterr().out
    assert "still working" in out
    assert "arch" in out
