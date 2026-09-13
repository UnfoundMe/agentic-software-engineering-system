"""`ToolSpec.allows_write_to` (docs/05 section 3.5)."""

from __future__ import annotations

from pathlib import PurePosixPath

from ases.kernel.tools.classification import SideEffect, ToolOutcome, ToolSpec


async def _noop_handler(args: object, ctx: object) -> ToolOutcome:
    return ToolOutcome(ok=True)


def _spec(**overrides: object) -> ToolSpec:
    defaults: dict[str, object] = {
        "name": "t",
        "handler": _noop_handler,
        "idempotent": True,
        "side_effect": SideEffect.SANDBOX,
    }
    defaults.update(overrides)
    return ToolSpec(**defaults)  # type: ignore[arg-type]


def test_a_readonly_tool_with_no_writable_paths_allows_nothing() -> None:
    spec = _spec(writable_paths=())
    assert spec.allows_write_to(PurePosixPath("anything.txt")) is False


def test_a_path_under_the_writable_root_is_allowed() -> None:
    spec = _spec(writable_paths=(PurePosixPath("."),))
    assert spec.allows_write_to(PurePosixPath("sub/dir/file.txt")) is True


def test_a_path_under_a_specific_writable_prefix_is_allowed() -> None:
    spec = _spec(writable_paths=(PurePosixPath("src"),))
    assert spec.allows_write_to(PurePosixPath("src/file.txt")) is True


def test_a_path_outside_every_writable_prefix_is_denied() -> None:
    spec = _spec(writable_paths=(PurePosixPath("src"),))
    assert spec.allows_write_to(PurePosixPath("docs/file.txt")) is False


def test_a_sibling_path_is_not_confused_with_a_prefix_match() -> None:
    """`src2/file.txt` must not be treated as being "under" `src` just
    because the string `src` is a leading substring."""
    spec = _spec(writable_paths=(PurePosixPath("src"),))
    assert spec.allows_write_to(PurePosixPath("src2/file.txt")) is False
