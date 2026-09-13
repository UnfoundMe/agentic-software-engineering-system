"""The `dotnet` subcommands docs/02 Phase 3 names - `new`, `restore`, `build`,
`test`, `format`, `list package --vulnerable` - plus four subcommands added
when a genuine live run (real LLM, real `dotnet` toolchain) surfaced gaps
`agents/scaffold.py` had disclosed or silently carried:

- `new sln` / `sln add` / `add reference`: without them, a real multi-project
  build fails outright because a later project cannot reference an earlier
  one's types at all.
- `add_package`: `SolutionSkeleton.pinned_packages` was being reported in the
  artifact but never actually installed into any `.csproj` - any code
  depending on one of those packages (EF Core, Npgsql, ...) would fail to
  compile for the same "the type doesn't exist" reason.

Both classes of failure have no fix a code-rewriting repair agent could ever
apply - they are project *structure*, not file *content*. Nothing more than
these ten - this is what "no unrestricted shell access" (CLAUDE.md section 6)
looks like in practice: fixed, named commands an agent can invoke, never an
arbitrary command line.

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

#: CLAUDE.md section 1 pins the stack to ".NET 10 / ASP.NET Core" - this is
#: not a moving target to detect from the installed SDK, it is this
#: project's own fixed technology choice. Passed explicitly to every
#: `dotnet new` call because different templates default to *different*
#: target frameworks otherwise: found live (`27a721ce-...`) when the
#: `webapi` template's own default (an LTS release, `net8.0` on this SDK)
#: landed a different `TargetFramework` than the `classlib` templates'
#: default (this SDK's own version, `net10.0`) in the very same solution -
#: `UrlShortener.Api` could not reference `UrlShortener.Application`
#: (NU1201/NU1603: incompatible target frameworks), a restore failure no
#: amount of `dotnet.add_reference` correctness could ever fix.
TARGET_FRAMEWORK = "net10.0"


def build_dotnet_tools(runner: SubprocessRunner = default_runner) -> tuple[ToolSpec, ...]:
    async def dotnet_new(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        name = str(args["name"])
        template = str(args["template"])
        argv = ["dotnet", "new", template, "-n", name, "-o", name, "-f", TARGET_FRAMEWORK]
        if template == "webapi":
            # ImplementerAgent's `impl_api` focus explicitly says "thin
            # controllers" (see `agents/implementer.py`'s `_FOCUS_BY_NODE_ID`)
            # - the default `webapi` template since .NET 6 is minimal-API-only
            # unless this flag is given, which would leave nothing for a
            # controller class to attach to.
            argv.append("-controllers")
        return await run_argv(runner, argv, ctx)

    async def dotnet_restore(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        return await run_argv(runner, ["dotnet", "restore"], ctx)

    async def dotnet_build(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        argv = ["dotnet", "build"]
        if project := args.get("project"):
            # `workflows/greenfield.yaml`'s `build_domain`/`build_api` gate one
            # implementation task each - without this, both fell back to
            # building the whole solution (whatever a bare `dotnet build`
            # resolves to in `ctx.cwd`), making the two nodes literally
            # redundant: same command, same directory, dispatched together by
            # `Scheduler._step`'s genuine `asyncio.gather` concurrency. See
            # `kernel/tools/process.py`'s per-cwd lock for the resulting
            # concurrent-MSBuild hazard this alone does not fix - scoping the
            # build reduces how much of the solution two nodes redundantly
            # rebuild, the lock is what makes doing so at the same time safe.
            argv.append(str(project))
        argv.append("-warnaserror")
        return await run_argv(runner, argv, ctx)

    async def dotnet_test(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        argv = ["dotnet", "test"]
        if project := args.get("project"):
            argv.append(str(project))
        return await run_argv(runner, argv, ctx)

    async def dotnet_format_verify(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        return await run_argv(runner, ["dotnet", "format", "--verify-no-changes"], ctx)

    async def dotnet_list_vulnerable(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        return await run_argv(runner, ["dotnet", "list", "package", "--vulnerable"], ctx)

    async def dotnet_new_sln(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        return await run_argv(runner, ["dotnet", "new", "sln", "-n", str(args["name"])], ctx)

    async def dotnet_sln_add(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        return await run_argv(runner, ["dotnet", "sln", "add", str(args["project"])], ctx)

    async def dotnet_add_reference(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        return await run_argv(
            runner,
            ["dotnet", "add", str(args["project"]), "reference", str(args["reference"])],
            ctx,
        )

    async def dotnet_add_package(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        argv = ["dotnet", "add", str(args["project"]), "package", str(args["package"])]
        version = args.get("version")
        if version:
            argv += ["--version", str(version)]
        return await run_argv(runner, argv, ctx)

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
            writable_paths=(PurePosixPath("."),),
            path_args=("project",),
        ),
        ToolSpec(
            name="dotnet.test",
            handler=dotnet_test,
            idempotent=True,
            side_effect=SideEffect.SANDBOX,
            timeout_s=300.0,
            writable_paths=(PurePosixPath("."),),
            path_args=("project",),
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
        ToolSpec(
            name="dotnet.new_sln",
            handler=dotnet_new_sln,
            idempotent=False,  # re-running with the same name fails on an existing .sln
            side_effect=SideEffect.SANDBOX,
            writable_paths=(PurePosixPath("."),),
            path_args=("name",),
        ),
        ToolSpec(
            name="dotnet.sln_add",
            handler=dotnet_sln_add,
            idempotent=True,  # adding an already-added project is a no-op
            side_effect=SideEffect.SANDBOX,
            writable_paths=(PurePosixPath("."),),
            path_args=("project",),
        ),
        ToolSpec(
            name="dotnet.add_reference",
            handler=dotnet_add_reference,
            idempotent=True,  # adding an already-added reference is a no-op
            side_effect=SideEffect.SANDBOX,
            writable_paths=(PurePosixPath("."),),
            path_args=("project", "reference"),
        ),
        ToolSpec(
            name="dotnet.add_package",
            handler=dotnet_add_package,
            idempotent=True,  # re-adding the same package/version is a no-op
            side_effect=SideEffect.SANDBOX,
            writable_paths=(PurePosixPath("."),),
            path_args=("project",),
            timeout_s=120.0,  # resolves against NuGet - slower than a local-only edit
        ),
    )
