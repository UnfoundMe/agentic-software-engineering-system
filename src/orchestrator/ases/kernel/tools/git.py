"""Git tools: read-only inspection only.

**Scoping note, stated explicitly rather than silently decided:** docs/02's
repository layout names this file but Phase 3's own bullet list never
specifies which git actions belong in it - unlike `dotnet.py`, where the six
subcommands are named exactly. Any git operation that changes history or a
ref (`commit`, `push`, `merge`, `checkout`) is already forbidden to agents by
CLAUDE.md section 13 ("must not push or merge without explicit
authorization... must not directly modify protected branches") and belongs,
if anywhere, to a human-gated promotion step - not an agent-invocable tool.
So this module registers only the two read-only operations that are
unambiguously safe and useful today (a reviewer inspecting what changed) and
adds nothing else. If a future agent needs a mutating git action, that is a
deliberate addition to make then, not an omission to fill in now.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ases.kernel.tools.classification import SideEffect, ToolContext, ToolOutcome, ToolSpec
from ases.kernel.tools.process import SubprocessRunner, default_runner, run_argv


def build_git_tools(runner: SubprocessRunner = default_runner) -> tuple[ToolSpec, ...]:
    async def git_status(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        return await run_argv(runner, ["git", "status", "--porcelain=v1"], ctx)

    async def git_diff(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        return await run_argv(runner, ["git", "diff"], ctx)

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
    )


__all__ = ["build_git_tools"]
