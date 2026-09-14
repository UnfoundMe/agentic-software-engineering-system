"""`kernel.tools.git`: read-only inspection, plus the one narrow
sandbox-reset exception (see the module docstring for why)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from ases.kernel.tools.classification import SideEffect, ToolContext
from ases.kernel.tools.git import build_git_tools


def _ctx(cwd: str = "/repo") -> ToolContext:
    return ToolContext(cwd=PurePosixPath(cwd), run_id="run-1")


class _FakeRunner:
    def __init__(self, returncode: int = 0, stdout: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.calls: list[tuple[Sequence[str], Path]] = []

    async def __call__(self, argv: Sequence[str], cwd: Path) -> tuple[int, str, str]:
        self.calls.append((argv, cwd))
        return self.returncode, self.stdout, ""


def test_exactly_three_tools_are_registered() -> None:
    names = {t.name for t in build_git_tools()}
    assert names == {"git.status", "git.diff", "git.reset_sandbox"}


def test_the_two_read_only_tools_declare_no_side_effect_and_no_writable_paths() -> None:
    for tool in build_git_tools():
        if tool.name == "git.reset_sandbox":
            continue
        assert tool.side_effect is SideEffect.NONE
        assert tool.writable_paths == ()
        assert tool.path_args == ()


def test_reset_sandbox_is_a_sandbox_scoped_tool_needing_no_approval() -> None:
    tool = next(t for t in build_git_tools() if t.name == "git.reset_sandbox")
    assert tool.side_effect is SideEffect.SANDBOX
    assert tool.requires_approval is False
    assert tool.idempotent is True
    assert tool.writable_paths == (PurePosixPath("."),)


async def test_git_status_invokes_the_correct_argv() -> None:
    runner = _FakeRunner(stdout=" M README.md\n")
    tool = next(t for t in build_git_tools(runner) if t.name == "git.status")
    result = await tool.handler({}, _ctx())
    assert result.ok is True
    assert result.output["stdout"] == " M README.md\n"
    argv, _ = runner.calls[0]
    assert argv == ["git", "status", "--porcelain=v1"]


async def test_git_diff_invokes_the_correct_argv() -> None:
    runner = _FakeRunner()
    tool = next(t for t in build_git_tools(runner) if t.name == "git.diff")
    await tool.handler({}, _ctx())
    argv, _ = runner.calls[0]
    assert argv == ["git", "diff"]


async def test_a_failing_git_invocation_is_reported() -> None:
    runner = _FakeRunner(returncode=128)
    tool = next(t for t in build_git_tools(runner) if t.name == "git.status")
    result = await tool.handler({}, _ctx())
    assert result.ok is False
    assert "exited 128" in (result.error or "")


async def test_reset_sandbox_resets_then_cleans_in_order() -> None:
    runner = _FakeRunner()
    tool = next(t for t in build_git_tools(runner) if t.name == "git.reset_sandbox")
    result = await tool.handler({}, _ctx())
    assert result.ok is True
    argv_calls = [argv for argv, _ in runner.calls]
    assert argv_calls == [
        ["git", "reset", "--hard", "HEAD"],
        ["git", "clean", "-xdf", "."],
    ]


async def test_reset_sandbox_never_cleans_if_the_reset_itself_fails() -> None:
    runner = _FakeRunner(returncode=1)
    tool = next(t for t in build_git_tools(runner) if t.name == "git.reset_sandbox")
    result = await tool.handler({}, _ctx())
    assert result.ok is False
    argv_calls = [argv for argv, _ in runner.calls]
    assert argv_calls == [["git", "reset", "--hard", "HEAD"]]
