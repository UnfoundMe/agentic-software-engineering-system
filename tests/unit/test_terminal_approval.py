"""`interfaces.terminal_approval.TerminalApprovalProvider` - the minimal,
real human-in-the-loop `ApprovalProvider` `ases run` uses.

`input()` is monkeypatched rather than driven through a real terminal; what's
under test is the wiring (artifact lookup, decision construction), not the
terminal itself.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from ases.contracts.artifacts import DesignSpec
from ases.interfaces.terminal_approval import TerminalApprovalProvider
from ases.kernel.graph import NodeKind, NodeSpec
from ases.kernel.state import ApprovalRecord, ArtifactRecord, RunState


def _gate_node(node_id: str = "gate2", description: str = "Sign off on the design.") -> NodeSpec:
    return NodeSpec(id=node_id, kind=NodeKind.GATE, requires_approval=True, description=description)


def _state_awaiting_approval(node_id: str = "gate2") -> RunState:
    state = RunState(run_id=uuid4())
    design = DesignSpec(summary="4-layer solution")
    state.artifacts["h-design"] = ArtifactRecord(
        artifact_hash="h-design", kind="DesignSpec", node_id="arch", produced_at=datetime.now(UTC)
    )
    state.artifact_content["h-design"] = design.model_dump(mode="json")
    state.approvals[node_id] = ApprovalRecord(
        node_id=node_id,
        artifact_hash="h-design",
        granted=False,
        actor="",
        decided_at=datetime.now(UTC),
    )
    return state


def _script_inputs(monkeypatch: pytest.MonkeyPatch, answers: list[str]) -> None:
    queue = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(queue))


async def test_decide_grants_on_y(monkeypatch: pytest.MonkeyPatch) -> None:
    _script_inputs(monkeypatch, ["y", ""])
    provider = TerminalApprovalProvider()

    decision = await provider.decide(_gate_node(), _state_awaiting_approval())

    assert decision is not None
    assert decision.granted is True
    assert decision.reason == ""


async def test_decide_rejects_on_anything_else(monkeypatch: pytest.MonkeyPatch) -> None:
    _script_inputs(monkeypatch, ["n", "needs another pass"])
    provider = TerminalApprovalProvider()

    decision = await provider.decide(_gate_node(), _state_awaiting_approval())

    assert decision is not None
    assert decision.granted is False
    assert decision.reason == "needs another pass"


async def test_decide_treats_empty_answer_as_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    _script_inputs(monkeypatch, ["", ""])
    provider = TerminalApprovalProvider()

    decision = await provider.decide(_gate_node(), _state_awaiting_approval())

    assert decision is not None
    assert decision.granted is False


async def test_decide_records_the_current_os_user_as_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    _script_inputs(monkeypatch, ["y", ""])
    monkeypatch.setattr("getpass.getuser", lambda: "alice")
    provider = TerminalApprovalProvider()

    decision = await provider.decide(_gate_node(), _state_awaiting_approval())

    assert decision is not None
    assert decision.actor == "alice"


async def test_decide_prints_the_pending_artifacts_content(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _script_inputs(monkeypatch, ["y", ""])
    provider = TerminalApprovalProvider()

    await provider.decide(_gate_node(), _state_awaiting_approval())

    out = capsys.readouterr().out
    assert "DesignSpec" in out
    assert "4-layer solution" in out


async def test_decide_handles_no_approval_record_gracefully(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _script_inputs(monkeypatch, ["y", ""])
    provider = TerminalApprovalProvider()

    decision = await provider.decide(_gate_node("gate1"), RunState(run_id=uuid4()))

    assert decision is not None
    assert decision.granted is True
    assert "no artifact hash recorded" in capsys.readouterr().out


async def test_decide_handles_unretrievable_artifact_content_gracefully(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A hash recorded in `state.artifacts` but with no `artifact_content`
    entry (the residual pre-Phase-4 case `ContextRetriever.fetch` still
    guards) must not crash the approval prompt itself."""
    _script_inputs(monkeypatch, ["y", ""])
    state = RunState(run_id=uuid4())
    state.artifacts["h-old"] = ArtifactRecord(
        artifact_hash="h-old", kind="DesignSpec", node_id="arch", produced_at=datetime.now(UTC)
    )
    state.approvals["gate2"] = ApprovalRecord(
        node_id="gate2",
        artifact_hash="h-old",
        granted=False,
        actor="",
        decided_at=datetime.now(UTC),
    )
    provider = TerminalApprovalProvider()

    decision = await provider.decide(_gate_node(), state)

    assert decision is not None
    assert "could not load artifact" in capsys.readouterr().out
