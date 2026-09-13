"""`kernel.tools.git`: read-only inspection only (see the module docstring
for why no mutating action is registered)."""

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


def test_only_two_read_only_tools_are_registered() -> None:
    names = {t.name for t in build_git_tools()}
    assert names == {"git.status", "git.diff"}


def test_both_git_tools_declare_no_side_effect_and_no_writable_paths() -> None:
    for tool in build_git_tools():
        assert tool.side_effect is SideEffect.NONE
        assert tool.writable_paths == ()
        assert tool.path_args == ()


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
