"""Shared subprocess-invocation plumbing for `dotnet.py` and `git.py`.

Both tool modules wrap a small, fixed set of named subcommands - never an
arbitrary command line (CLAUDE.md section 6) - and both need the same
"run this argv, capture output, turn it into a `ToolOutcome`" shape. Factored
here once so it is not duplicated, and so a handler's own logic (which argv
to build from `args`) stays the only thing that differs between tools.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

from ases.kernel.tools.classification import ToolContext, ToolOutcome

#: (argv, cwd) -> (returncode, stdout, stderr). The default runs a real
#: subprocess; tests inject a fake one so no external binary is required.
SubprocessRunner = Callable[[Sequence[str], Path], Awaitable[tuple[int, str, str]]]


async def default_runner(argv: Sequence[str], cwd: Path) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def run_argv(runner: SubprocessRunner, argv: Sequence[str], ctx: ToolContext) -> ToolOutcome:
    returncode, stdout, stderr = await runner(argv, Path(ctx.cwd))
    return ToolOutcome(
        ok=returncode == 0,
        output={"returncode": returncode, "stdout": stdout, "stderr": stderr},
        error=None if returncode == 0 else f"`{' '.join(argv)}` exited {returncode}",
    )
