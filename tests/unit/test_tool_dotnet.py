"""`kernel.tools.dotnet` (docs/02 Phase 3's six named subcommands, plus the
three solution-linking ones added when a live run surfaced the need for
them - see the module's own docstring).

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


def test_exactly_the_ten_named_subcommands_are_registered() -> None:
    names = {t.name for t in build_dotnet_tools()}
    assert names == {
        "dotnet.new",
        "dotnet.restore",
        "dotnet.build",
        "dotnet.test",
        "dotnet.format_verify",
        "dotnet.list_vulnerable",
        "dotnet.new_sln",
        "dotnet.sln_add",
        "dotnet.add_reference",
        "dotnet.add_package",
    }


async def test_dotnet_new_invokes_the_correct_argv_for_classlib() -> None:
    runner = _FakeRunner()
    tool = _tool("dotnet.new", runner)
    result = await tool.handler({"template": "classlib", "name": "UrlShortener.Domain"}, _ctx())
    assert result.ok is True
    argv, _ = runner.calls[0]
    assert argv == [
        "dotnet",
        "new",
        "classlib",
        "-n",
        "UrlShortener.Domain",
        "-o",
        "UrlShortener.Domain",
        "-f",
        "net10.0",
    ]


async def test_dotnet_new_appends_controllers_flag_for_webapi_only() -> None:
    """`impl_api`'s focus is explicitly "thin controllers" (see
    `agents/implementer.py`) - the default `webapi` template is minimal-API
    only unless this flag is present."""
    runner = _FakeRunner()
    tool = _tool("dotnet.new", runner)
    await tool.handler({"template": "webapi", "name": "UrlShortener.Api"}, _ctx())
    argv, _ = runner.calls[0]
    assert argv == [
        "dotnet",
        "new",
        "webapi",
        "-n",
        "UrlShortener.Api",
        "-o",
        "UrlShortener.Api",
        "-f",
        "net10.0",
        "-controllers",
    ]


async def test_dotnet_new_always_pins_the_target_framework_regardless_of_template() -> None:
    """Found live (`27a721ce-...`): `webapi` and `classlib` templates default
    to *different* target frameworks on the same SDK, so a solution scaffolded
    without this flag could not restore (`UrlShortener.Api` on net8.0
    referencing `UrlShortener.Application` on net10.0 - NU1201/NU1603).
    Every project must get the identical, explicitly pinned framework."""
    for template in ("classlib", "webapi"):
        runner = _FakeRunner()
        tool = _tool("dotnet.new", runner)
        await tool.handler({"template": template, "name": "X"}, _ctx())
        argv, _ = runner.calls[0]
        assert "-f" in argv
        assert argv[argv.index("-f") + 1] == "net10.0"


async def test_dotnet_build_passes_warnaserror() -> None:
    runner = _FakeRunner()
    tool = _tool("dotnet.build", runner)
    await tool.handler({}, _ctx())
    argv, _ = runner.calls[0]
    assert argv == ["dotnet", "build", "-warnaserror"]


async def test_dotnet_build_scopes_to_a_project_when_given() -> None:
    """`build_domain`/`build_api` (`agents/wiring.py`'s `_build_domain_args`/
    `_build_api_args`) pass this so the two nodes stop being the same
    unscoped whole-solution build - see docs/07 issue #17."""
    runner = _FakeRunner()
    tool = _tool("dotnet.build", runner)
    await tool.handler({"project": "UrlShortener.Domain"}, _ctx())
    argv, _ = runner.calls[0]
    assert argv == ["dotnet", "build", "UrlShortener.Domain", "-warnaserror"]


async def test_dotnet_test_scopes_to_a_project_when_given() -> None:
    runner = _FakeRunner()
    tool = _tool("dotnet.test", runner)
    await tool.handler({"project": "UrlShortener.Tests"}, _ctx())
    argv, _ = runner.calls[0]
    assert argv == ["dotnet", "test", "UrlShortener.Tests"]


def test_build_and_test_declare_the_optional_project_path_arg() -> None:
    tools = {t.name: t for t in build_dotnet_tools()}
    assert tools["dotnet.build"].path_args == ("project",)
    assert tools["dotnet.test"].path_args == ("project",)


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
    # Not just the exit code: `kernel.tools.process.run_argv` folds the actual
    # diagnostic into `.error` too, since that is the only thing a repair
    # agent ever sees back (`ContextRetriever.last_error_of` reads this exact
    # field) - an exit code alone gives it nothing to fix.
    assert "CS0103: name does not exist" in (result.error or "")


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


def test_new_and_new_sln_are_the_only_non_idempotent_tools() -> None:
    tools = {t.name: t for t in build_dotnet_tools()}
    non_idempotent = {"dotnet.new", "dotnet.new_sln"}
    assert {name for name, t in tools.items() if not t.idempotent} == non_idempotent


def test_new_declares_its_writable_path_and_path_arg() -> None:
    tools = {t.name: t for t in build_dotnet_tools()}
    assert tools["dotnet.new"].path_args == ("name",)
    assert tools["dotnet.new"].writable_paths == (PurePosixPath("."),)


async def test_dotnet_new_sln_invokes_the_correct_argv() -> None:
    runner = _FakeRunner()
    tool = _tool("dotnet.new_sln", runner)
    await tool.handler({"name": "UrlShortener"}, _ctx())
    argv, _ = runner.calls[0]
    assert argv == ["dotnet", "new", "sln", "-n", "UrlShortener"]


async def test_dotnet_sln_add_invokes_the_correct_argv() -> None:
    runner = _FakeRunner()
    tool = _tool("dotnet.sln_add", runner)
    await tool.handler({"project": "UrlShortener.Domain"}, _ctx())
    argv, _ = runner.calls[0]
    assert argv == ["dotnet", "sln", "add", "UrlShortener.Domain"]


async def test_dotnet_add_reference_invokes_the_correct_argv() -> None:
    runner = _FakeRunner()
    tool = _tool("dotnet.add_reference", runner)
    await tool.handler({"project": "UrlShortener.Api", "reference": "UrlShortener.Domain"}, _ctx())
    argv, _ = runner.calls[0]
    assert argv == ["dotnet", "add", "UrlShortener.Api", "reference", "UrlShortener.Domain"]


def test_sln_tools_declare_their_writable_path_and_path_args() -> None:
    tools = {t.name: t for t in build_dotnet_tools()}
    assert tools["dotnet.new_sln"].path_args == ("name",)
    assert tools["dotnet.sln_add"].path_args == ("project",)
    assert tools["dotnet.add_reference"].path_args == ("project", "reference")
    for name in ("dotnet.new_sln", "dotnet.sln_add", "dotnet.add_reference"):
        assert tools[name].writable_paths == (PurePosixPath("."),)


def test_sln_add_and_add_reference_are_idempotent_new_sln_is_not() -> None:
    tools = {t.name: t for t in build_dotnet_tools()}
    assert tools["dotnet.new_sln"].idempotent is False
    assert tools["dotnet.sln_add"].idempotent is True
    assert tools["dotnet.add_reference"].idempotent is True


async def test_dotnet_add_package_invokes_the_correct_argv_with_a_version() -> None:
    runner = _FakeRunner()
    tool = _tool("dotnet.add_package", runner)
    await tool.handler(
        {
            "project": "UrlShortener.Infrastructure",
            "package": "Npgsql.EntityFrameworkCore.PostgreSQL",
            "version": "10.0.0",
        },
        _ctx(),
    )
    argv, _ = runner.calls[0]
    assert argv == [
        "dotnet",
        "add",
        "UrlShortener.Infrastructure",
        "package",
        "Npgsql.EntityFrameworkCore.PostgreSQL",
        "--version",
        "10.0.0",
    ]


async def test_dotnet_add_package_omits_version_flag_when_not_given() -> None:
    runner = _FakeRunner()
    tool = _tool("dotnet.add_package", runner)
    await tool.handler({"project": "UrlShortener.Api", "package": "Serilog"}, _ctx())
    argv, _ = runner.calls[0]
    assert argv == ["dotnet", "add", "UrlShortener.Api", "package", "Serilog"]


def test_add_package_declares_its_writable_path_and_path_arg() -> None:
    tools = {t.name: t for t in build_dotnet_tools()}
    assert tools["dotnet.add_package"].path_args == ("project",)
    assert tools["dotnet.add_package"].writable_paths == (PurePosixPath("."),)
    assert tools["dotnet.add_package"].idempotent is True
    assert tools["dotnet.add_package"].side_effect is SideEffect.SANDBOX
