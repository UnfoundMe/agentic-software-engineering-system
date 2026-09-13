"""The exact six `dotnet` subcommands docs/02 Phase 3 names: `new`, `restore`,
`build`, `test`, `format`, `list package --vulnerable`. Nothing more - this is
what "no unrestricted shell access" (CLAUDE.md section 6) looks like in
practice: six fixed, named commands an agent can invoke, never an arbitrary
command line.

`build_dotnet_tools` takes the subprocess runner as a parameter so tests can
supply a fake one and never actually shell out to a `dotnet` binary - the
handlers themselves are exercised for real, only the process boundary is
substituted (docs/06's "never claim success without actual evidence" cuts
both ways: a test must not need a real toolchain installed to prove the
*wiring* is correct).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from ases.kernel.tools.classification import SideEffect, ToolContext, ToolOutcome, ToolSpec
from ases.kernel.tools.process import SubprocessRunner, default_runner, run_argv


def build_dotnet_tools(runner: SubprocessRunner = default_runner) -> tuple[ToolSpec, ...]:
    async def dotnet_new(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        name = str(args["name"])
        return await run_argv(
            runner, ["dotnet", "new", str(args["template"]), "-n", name, "-o", name], ctx
        )

    async def dotnet_restore(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        return await run_argv(runner, ["dotnet", "restore"], ctx)

    async def dotnet_build(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        return await run_argv(runner, ["dotnet", "build", "-warnaserror"], ctx)

    async def dotnet_test(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        return await run_argv(runner, ["dotnet", "test"], ctx)

    async def dotnet_format_verify(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        return await run_argv(runner, ["dotnet", "format", "--verify-no-changes"], ctx)

    async def dotnet_list_vulnerable(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        return await run_argv(runner, ["dotnet", "list", "package", "--vulnerable"], ctx)

    return (
        ToolSpec(
            name="dotnet.new",
            handler=dotnet_new,
            idempotent=False,  # re-running without --force fails on an existing project
            side_effect=SideEffect.SANDBOX,
            writable_paths=(PurePosixPath("."),),
            path_args=("name",),
        ),
        ToolSpec(
            name="dotnet.restore",
            handler=dotnet_restore,
            idempotent=True,
            side_effect=SideEffect.SANDBOX,
        ),
        ToolSpec(
            name="dotnet.build",
            handler=dotnet_build,
            idempotent=True,
            side_effect=SideEffect.SANDBOX,
            timeout_s=300.0,
        ),
        ToolSpec(
            name="dotnet.test",
            handler=dotnet_test,
            idempotent=True,
            side_effect=SideEffect.SANDBOX,
            timeout_s=300.0,
        ),
        ToolSpec(
            name="dotnet.format_verify",
            handler=dotnet_format_verify,
            idempotent=True,
            side_effect=SideEffect.NONE,  # `--verify-no-changes` never writes
        ),
        ToolSpec(
            name="dotnet.list_vulnerable",
            handler=dotnet_list_vulnerable,
            idempotent=True,
            side_effect=SideEffect.NONE,
        ),
    )
