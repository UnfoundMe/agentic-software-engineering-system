"""`kernel.tools.dotnet` (docs/02 Phase 3: exactly the six named subcommands).

Every test injects a fake subprocess runner - no real `dotnet` binary is
required, and the suite must pass identically whether or not the .NET SDK is
installed on the machine running it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from ases.kernel.tools.classification import SideEffect, ToolContext
from ases.kernel.tools.dotnet import build_dotnet_tools


def _ctx(cwd: str = "/sandbox") -> ToolContext:
    return ToolContext(cwd=PurePosixPath(cwd), run_id="run-1")


class _FakeRunner:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.calls: list[tuple[Sequence[str], Path]] = []

    async def __call__(self, argv: Sequence[str], cwd: Path) -> tuple[int, str, str]:
        self.calls.append((argv, cwd))
        return self.returncode, self.stdout, self.stderr


def _tool(name: str, runner: _FakeRunner):  # type: ignore[no-untyped-def]
    tools = build_dotnet_tools(runner)
    return next(t for t in tools if t.name == name)


def test_exactly_the_six_named_subcommands_are_registered() -> None:
    names = {t.name for t in build_dotnet_tools()}
    assert names == {
        "dotnet.new",
        "dotnet.restore",
        "dotnet.build",
        "dotnet.test",
        "dotnet.format_verify",
        "dotnet.list_vulnerable",
    }


async def test_dotnet_new_invokes_the_correct_argv() -> None:
    runner = _FakeRunner()
    tool = _tool("dotnet.new", runner)
    result = await tool.handler({"template": "webapi", "name": "UrlShortener"}, _ctx())
    assert result.ok is True
    argv, _ = runner.calls[0]
    assert argv == ["dotnet", "new", "webapi", "-n", "UrlShortener", "-o", "UrlShortener"]


async def test_dotnet_build_passes_warnaserror() -> None:
    runner = _FakeRunner()
    tool = _tool("dotnet.build", runner)
    await tool.handler({}, _ctx())
    argv, _ = runner.calls[0]
    assert argv == ["dotnet", "build", "-warnaserror"]


async def test_dotnet_format_verify_uses_verify_no_changes() -> None:
    runner = _FakeRunner()
    tool = _tool("dotnet.format_verify", runner)
    await tool.handler({}, _ctx())
    argv, _ = runner.calls[0]
    assert argv == ["dotnet", "format", "--verify-no-changes"]


async def test_dotnet_list_vulnerable_argv() -> None:
    runner = _FakeRunner()
    tool = _tool("dotnet.list_vulnerable", runner)
    await tool.handler({}, _ctx())
    argv, _ = runner.calls[0]
    assert argv == ["dotnet", "list", "package", "--vulnerable"]


async def test_a_nonzero_exit_code_is_reported_as_failure_with_stderr_attached() -> None:
    runner = _FakeRunner(returncode=1, stderr="CS0103: name does not exist")
    tool = _tool("dotnet.build", runner)
    result = await tool.handler({}, _ctx())
    assert result.ok is False
    assert result.output["stderr"] == "CS0103: name does not exist"
    assert "exited 1" in (result.error or "")


async def test_the_runner_is_invoked_with_the_sandbox_cwd() -> None:
    runner = _FakeRunner()
    tool = _tool("dotnet.test", runner)
    await tool.handler({}, _ctx(cwd="/my/sandbox/root"))
    _, cwd = runner.calls[0]
    assert cwd == Path("/my/sandbox/root")


def test_build_and_test_have_longer_timeouts_than_the_cheap_checks() -> None:
    tools = {t.name: t for t in build_dotnet_tools()}
    assert tools["dotnet.build"].timeout_s > tools["dotnet.format_verify"].timeout_s
    assert tools["dotnet.test"].timeout_s > tools["dotnet.list_vulnerable"].timeout_s


def test_format_verify_and_list_vulnerable_are_side_effect_free() -> None:
    tools = {t.name: t for t in build_dotnet_tools()}
    assert tools["dotnet.format_verify"].side_effect is SideEffect.NONE
    assert tools["dotnet.list_vulnerable"].side_effect is SideEffect.NONE


def test_new_is_the_only_non_idempotent_tool() -> None:
    tools = {t.name: t for t in build_dotnet_tools()}
    assert tools["dotnet.new"].idempotent is False
    assert all(t.idempotent for name, t in tools.items() if name != "dotnet.new")


def test_new_declares_its_writable_path_and_path_arg() -> None:
    tools = {t.name: t for t in build_dotnet_tools()}
    assert tools["dotnet.new"].path_args == ("name",)
    assert tools["dotnet.new"].writable_paths == (PurePosixPath("."),)
