"""Git tools: read-only inspection, plus one narrow sandbox-reset exception.

**Scoping note, stated explicitly rather than silently decided:** docs/02's
repository layout names this file but Phase 3's own bullet list never
specifies which git actions belong in it - unlike `dotnet.py`, where the six
subcommands are named exactly. Any git operation that changes history or a
ref (`commit`, `push`, `merge`, `checkout` of a branch) is already forbidden
to agents by CLAUDE.md section 13 ("must not push or merge without explicit
authorization... must not directly modify protected branches") and belongs,
if anywhere, to a human-gated promotion step - not an agent-invocable tool.
So this module registers the two read-only operations that are unambiguously
safe and useful (a reviewer inspecting what changed), plus `git.reset_sandbox`
- a working-tree-only reset (never a ref/history change) that lets a retried
`agents/scaffold.py` attempt start from the same clean state the first
attempt did, instead of colliding with whatever a non-idempotent `dotnet new`/
`dotnet new sln` already wrote. Nothing else is added. If a future agent
needs another mutating git action, that is a deliberate addition to make
then, not an omission to fill in now.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from ases.kernel.tools.classification import SideEffect, ToolContext, ToolOutcome, ToolSpec
from ases.kernel.tools.process import SubprocessRunner, default_runner, run_argv


def build_git_tools(runner: SubprocessRunner = default_runner) -> tuple[ToolSpec, ...]:
    async def git_status(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        return await run_argv(runner, ["git", "status", "--porcelain=v1"], ctx)

    async def git_diff(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        return await run_argv(runner, ["git", "diff"], ctx)

    async def git_reset_sandbox(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        """Discards every uncommitted change in the sandbox worktree, tracked
        and untracked alike, restoring it to the commit it was checked out
        from. This is a mutating exception to the module's read-only rule
        (see the module docstring), deliberately narrow: unlike `checkout`/
        `commit`/`push`/`merge`, it never touches history or a ref, only the
        working tree of a sandbox `agents/scaffold.py` may already have
        partially populated on a prior, failed attempt - so a retry starts
        from the same clean state the first attempt did, instead of colliding
        with whatever `dotnet new`/`dotnet new sln` already wrote (neither is
        idempotent). Confined entirely to `ctx.cwd` (`SideEffect.SANDBOX`, the
        sandbox worktree, never the real repository - see `sandbox/workspace.py`),
        so it needs no more approval than the `dotnet.*` tools already granted
        to the same agent without one.
        """
        reset = await run_argv(runner, ["git", "reset", "--hard", "HEAD"], ctx)
        if not reset.ok:
            return reset
        return await run_argv(runner, ["git", "clean", "-xdf", "."], ctx)

    return (
        ToolSpec(
            name="git.status",
            handler=git_status,
            idempotent=True,
            side_effect=SideEffect.NONE,
        ),
        ToolSpec(
            name="git.diff",
            handler=git_diff,
            idempotent=True,
            side_effect=SideEffect.NONE,
        ),
        ToolSpec(
            name="git.reset_sandbox",
            handler=git_reset_sandbox,
            idempotent=True,
            side_effect=SideEffect.SANDBOX,
            requires_approval=False,
            writable_paths=(PurePosixPath("."),),
        ),
    )


__all__ = ["build_git_tools"]
