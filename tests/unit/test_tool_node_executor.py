"""`ToolNodeExecutor`: the `NodeExecutor` adapter for `kind: tool` graph
nodes (docs/03 section 3.3: `test_run`, `sec_scan`)."""

from __future__ import annotations

from pathlib import PurePosixPath
from uuid import uuid4

from ases.agents.tool_executor import ToolNodeExecutor
from ases.kernel.graph import NodeKind, NodeSpec
from ases.kernel.state import RunState
from ases.kernel.tools.classification import SideEffect, ToolOutcome, ToolSpec
from ases.kernel.tools.registry import ToolRegistry


def _node(node_id: str = "tool_node") -> NodeSpec:
    return NodeSpec(id=node_id, kind=NodeKind.TOOL, handler=node_id)


async def _ok_handler(args: object, ctx: object) -> ToolOutcome:
    return ToolOutcome(ok=True, output={"findings": []})


async def _failing_handler(args: object, ctx: object) -> ToolOutcome:
    return ToolOutcome(ok=False, error="build failed")


def _registry(name: str, handler: object) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name=name, handler=handler, idempotent=True, side_effect=SideEffect.NONE)  # type: ignore[arg-type]
    )
    return registry


async def test_execute_reports_success_with_no_artifact_by_default() -> None:
    executor = ToolNodeExecutor(
        "dotnet.test", tools=_registry("dotnet.test", _ok_handler), tool_cwd=PurePosixPath("/sb")
    )
    outcome = await executor.execute(_node(), RunState(run_id=uuid4()))
    assert outcome.ok is True
    assert outcome.artifact_kind is None


async def test_execute_reports_failure_with_the_tools_error_message() -> None:
    executor = ToolNodeExecutor(
        "dotnet.test",
        tools=_registry("dotnet.test", _failing_handler),
        tool_cwd=PurePosixPath("/sb"),
    )
    outcome = await executor.execute(_node(), RunState(run_id=uuid4()))
    assert outcome.ok is False
    assert outcome.error == "build failed"


async def test_execute_denies_an_unregistered_tool_as_a_failed_outcome() -> None:
    executor = ToolNodeExecutor("no.such.tool", tools=ToolRegistry(), tool_cwd=PurePosixPath("/sb"))
    outcome = await executor.execute(_node(), RunState(run_id=uuid4()))
    assert outcome.ok is False
    assert "denied" in (outcome.error or "")


async def test_build_args_is_consulted_and_passed_to_the_tool() -> None:
    captured: dict[str, object] = {}

    async def _capturing_handler(args: object, ctx: object) -> ToolOutcome:
        captured["args"] = args
        return ToolOutcome(ok=True)

    executor = ToolNodeExecutor(
        "scan",
        tools=_registry("scan", _capturing_handler),
        tool_cwd=PurePosixPath("/sb"),
        build_args=lambda retriever, node: {"content": "some code"},
    )
    await executor.execute(_node(), RunState(run_id=uuid4()))
    assert captured["args"] == {"content": "some code"}


async def test_build_args_sees_the_node_it_is_building_for() -> None:
    """One registered handler serves every dynamically admitted build
    node, so the args builder has to be able to tell them apart - it
    resolves each node's project from the task that node belongs to."""
    captured: dict[str, object] = {}

    async def _capturing_handler(args: object, ctx: object) -> ToolOutcome:
        captured["args"] = args
        return ToolOutcome(ok=True)

    executor = ToolNodeExecutor(
        "scan",
        tools=_registry("scan", _capturing_handler),
        tool_cwd=PurePosixPath("/sb"),
        build_args=lambda retriever, node: {"project": node.id},
    )
    await executor.execute(_node(), RunState(run_id=uuid4()))
    assert captured["args"] == {"project": _node().id}


async def test_build_artifact_produces_a_tracked_artifact_when_findings_exist() -> None:
    async def _scan_handler(args: object, ctx: object) -> ToolOutcome:
        return ToolOutcome(ok=False, output={"findings": [{"rule": "aws_key"}]}, error="1 found")

    executor = ToolNodeExecutor(
        "scan",
        tools=_registry("scan", _scan_handler),
        tool_cwd=PurePosixPath("/sb"),
        artifact_kind="PolicyViolation",
        build_artifact=lambda output: (
            {"rule": "secrets_detected", "message": str(output)} if output.get("findings") else None
        ),
    )
    outcome = await executor.execute(_node(), RunState(run_id=uuid4()))
    assert outcome.ok is False
    assert outcome.artifact_kind == "PolicyViolation"
    assert outcome.artifact_payload is not None


async def test_build_artifact_produces_nothing_on_a_clean_result() -> None:
    executor = ToolNodeExecutor(
        "scan",
        tools=_registry("scan", _ok_handler),
        tool_cwd=PurePosixPath("/sb"),
        artifact_kind="PolicyViolation",
        build_artifact=lambda output: {"rule": "x"} if output.get("findings") else None,
    )
    outcome = await executor.execute(_node(), RunState(run_id=uuid4()))
    assert outcome.ok is True
    assert outcome.artifact_kind is None
    assert outcome.artifact_payload is None
