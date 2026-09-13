"""`ToolRegistry` (docs/05 section 3.5): deny-by-default, unknown tool = DENY,
write outside `writable_paths` = DENY, and timeout enforcement.

**Design note on "unknown action":** the doc's phrasing is "unknown tool ->
DENY, unknown action on a known tool -> DENY." This registry has no
sub-action dispatch within one `ToolSpec` - each operation (`dotnet.build`,
`dotnet.test`, ...) is its own fully-qualified, separately registered tool
name, each with its own idempotency/timeout metadata. There is therefore no
"known tool, unknown action" case to construct: every name is either a fully
registered tool or unknown. This is a deliberate simplification, not a gap -
see `kernel/tools/registry.py`'s module docstring.
"""

from __future__ import annotations

from pathlib import PurePosixPath

import pytest

from ases.kernel.tools.classification import SideEffect, ToolContext, ToolOutcome, ToolSpec
from ases.kernel.tools.registry import InvocationVerdict, ToolRegistry, UnknownToolNameError


def _ctx(cwd: str = "/sandbox") -> ToolContext:
    return ToolContext(cwd=PurePosixPath(cwd), run_id="run-1")


async def _ok_handler(args: object, ctx: object) -> ToolOutcome:
    return ToolOutcome(ok=True, output={"echo": args})


async def _failing_handler(args: object, ctx: object) -> ToolOutcome:
    return ToolOutcome(ok=False, error="handler-reported failure")


async def _hanging_handler(args: object, ctx: object) -> ToolOutcome:
    import asyncio

    await asyncio.sleep(10)
    return ToolOutcome(ok=True)  # pragma: no cover - never reached


def _registry_with(*specs: ToolSpec) -> ToolRegistry:
    registry = ToolRegistry()
    for spec in specs:
        registry.register(spec)
    return registry


async def test_invoking_an_unregistered_tool_is_denied() -> None:
    registry = ToolRegistry()
    result = await registry.invoke("nonexistent.tool", {}, _ctx())
    assert result.verdict is InvocationVerdict.DENIED_UNKNOWN_TOOL
    assert result.denied is True
    assert result.ok is False


async def test_a_registered_readonly_tool_executes() -> None:
    spec = ToolSpec(name="echo", handler=_ok_handler, idempotent=True, side_effect=SideEffect.NONE)
    registry = _registry_with(spec)
    result = await registry.invoke("echo", {"x": 1}, _ctx())
    assert result.verdict is InvocationVerdict.EXECUTED
    assert result.ok is True
    assert result.output == {"echo": {"x": 1}}


async def test_a_failing_handler_is_still_executed_not_denied() -> None:
    """A handler-level failure is a different outcome from a registry-level
    denial - the call was permitted, it just didn't succeed."""
    spec = ToolSpec(
        name="fails", handler=_failing_handler, idempotent=True, side_effect=SideEffect.NONE
    )
    registry = _registry_with(spec)
    result = await registry.invoke("fails", {}, _ctx())
    assert result.verdict is InvocationVerdict.EXECUTED
    assert result.ok is False
    assert result.denied is False


async def test_a_write_within_the_writable_root_is_permitted() -> None:
    spec = ToolSpec(
        name="write",
        handler=_ok_handler,
        idempotent=True,
        side_effect=SideEffect.SANDBOX,
        writable_paths=(PurePosixPath("."),),
        path_args=("path",),
    )
    registry = _registry_with(spec)
    result = await registry.invoke("write", {"path": "sub/file.txt"}, _ctx())
    assert result.verdict is InvocationVerdict.EXECUTED
    assert result.ok is True


async def test_a_write_outside_the_writable_root_is_denied_and_the_handler_never_runs() -> None:
    calls: list[object] = []

    async def _tracking_handler(args: object, ctx: object) -> ToolOutcome:
        calls.append(args)
        return ToolOutcome(ok=True)

    spec = ToolSpec(
        name="write",
        handler=_tracking_handler,
        idempotent=True,
        side_effect=SideEffect.SANDBOX,
        writable_paths=(PurePosixPath("src"),),
        path_args=("path",),
    )
    registry = _registry_with(spec)
    result = await registry.invoke("write", {"path": "docs/file.txt"}, _ctx())

    assert result.verdict is InvocationVerdict.DENIED_WRITE_OUTSIDE_SANDBOX
    assert result.denied is True
    assert calls == []  # the handler must never have been called


async def test_a_path_traversal_attempt_is_denied_even_with_a_dot_writable_root() -> None:
    """Regression test for a real gap found while building this: `PurePosixPath`
    is purely lexical and never resolves `..`, so
    `PurePosixPath("../etc/passwd").is_relative_to(".")` is `True`. The
    registry must reject any `..` segment outright, before consulting
    `allows_write_to` at all."""
    spec = ToolSpec(
        name="write",
        handler=_ok_handler,
        idempotent=True,
        side_effect=SideEffect.SANDBOX,
        writable_paths=(PurePosixPath("."),),
        path_args=("path",),
    )
    registry = _registry_with(spec)
    result = await registry.invoke("write", {"path": "../etc/passwd"}, _ctx())
    assert result.verdict is InvocationVerdict.DENIED_WRITE_OUTSIDE_SANDBOX


async def test_a_windows_backslash_absolute_path_is_denied_at_the_registry_too() -> None:
    """The same confirmed exploit as `test_tool_fs.py`'s regression test,
    checked at the registry's structural enforcement point rather than
    inside one handler - both layers must independently refuse it."""
    spec = ToolSpec(
        name="write",
        handler=_ok_handler,
        idempotent=True,
        side_effect=SideEffect.SANDBOX,
        writable_paths=(PurePosixPath("."),),
        path_args=("path",),
    )
    registry = _registry_with(spec)
    result = await registry.invoke("write", {"path": r"C:\Windows\System32\evil.txt"}, _ctx())
    assert result.verdict is InvocationVerdict.DENIED_WRITE_OUTSIDE_SANDBOX


async def test_an_absolute_path_is_denied_regardless_of_writable_paths() -> None:
    spec = ToolSpec(
        name="write",
        handler=_ok_handler,
        idempotent=True,
        side_effect=SideEffect.SANDBOX,
        writable_paths=(PurePosixPath("."),),
        path_args=("path",),
    )
    registry = _registry_with(spec)
    result = await registry.invoke("write", {"path": "/etc/passwd"}, _ctx())
    assert result.verdict is InvocationVerdict.DENIED_WRITE_OUTSIDE_SANDBOX


async def test_a_tool_that_exceeds_its_timeout_is_reported_as_timed_out() -> None:
    spec = ToolSpec(
        name="slow",
        handler=_hanging_handler,
        idempotent=True,
        side_effect=SideEffect.NONE,
        timeout_s=0.05,
    )
    registry = _registry_with(spec)
    result = await registry.invoke("slow", {}, _ctx())
    assert result.verdict is InvocationVerdict.DENIED_TIMED_OUT
    assert result.ok is False


def test_registering_the_same_name_twice_raises() -> None:
    spec = ToolSpec(name="dup", handler=_ok_handler, idempotent=True, side_effect=SideEffect.NONE)
    registry = _registry_with(spec)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec)


def test_spec_for_an_unknown_tool_raises() -> None:
    registry = ToolRegistry()
    with pytest.raises(UnknownToolNameError):
        registry.spec_for("nonexistent")


def test_known_tools_lists_every_registered_name_sorted() -> None:
    registry = _registry_with(
        ToolSpec(name="b", handler=_ok_handler, idempotent=True, side_effect=SideEffect.NONE),
        ToolSpec(name="a", handler=_ok_handler, idempotent=True, side_effect=SideEffect.NONE),
    )
    assert registry.known_tools() == ("a", "b")
