"""Shared subprocess-invocation plumbing for `dotnet.py` and `git.py`.

Both tool modules wrap a small, fixed set of named subcommands - never an
arbitrary command line (CLAUDE.md section 6) - and both need the same
"run this argv, capture output, turn it into a `ToolOutcome`" shape. Factored
here once so it is not duplicated, and so a handler's own logic (which argv
to build from `args`) stays the only thing that differs between tools.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

from ases.kernel.tools.classification import ToolContext, ToolOutcome

#: Found live (`60ed0166-...`): `dotnet add package` crashed deep inside
#: NuGet.targets while writing the dependency-graph spec (a
#: `Newtonsoft.Json.JsonWriter` static-constructor failure), immediately
#: after two prior `dotnet` invocations (`dotnet new`, `dotnet sln add`)
#: against the same project moments earlier. This is a documented MSBuild
#: node-reuse defect: the SDK keeps a worker process alive between `dotnet`
#: CLI invocations for speed, and that process's state can become corrupted
#: under rapid sequential invocations against the same project - exactly
#: this codebase's pattern (`agents/scaffold.py` calls `dotnet.new` ->
#: `dotnet.sln_add` -> `dotnet.add_package` back to back for every project).
#: `MSBUILDDISABLENODEREUSE=1` forces a fresh MSBuild process per invocation,
#: eliminating the stale-node-state class of failure entirely. Harmless to
#: set for `git` invocations too (this env var means nothing to git).
_SUBPROCESS_ENV = {**os.environ, "MSBUILDDISABLENODEREUSE": "1"}

#: (argv, cwd) -> (returncode, stdout, stderr). The default runs a real
#: subprocess; tests inject a fake one so no external binary is required.
SubprocessRunner = Callable[[Sequence[str], Path], Awaitable[tuple[int, str, str]]]

#: Found live (`b5da55c3-...`): `workflows/greenfield.yaml`'s `build_domain`
#: and `build_api` become ready in the same scheduling pass and are
#: dispatched together via `asyncio.gather` (`kernel.scheduler.Scheduler._step`
#: - genuinely concurrent by design). Both invoke `dotnet build` against the
#: same solution directory; two real, simultaneous MSBuild processes writing
#: to the same `obj`/`bin` trees is a second, distinct concurrency hazard from
#: #9's node-reuse defect above (that one is about *sequential* invocations
#: corrupting a reused worker process's state; this one is about two
#: processes racing on the same files at the same instant) - it produced two
#: divergent, non-reproducible failures on the retry (an `MSB4018` file-handle
#: crash for one, an otherwise-inexplicable `CS0234` missing-type error for
#: the other) from what should have been the same deterministic build.
#: Serializing here, keyed by the working directory every `dotnet`/`git`
#: invocation actually runs in, removes the hazard at its source regardless
#: of which graph nodes happen to trigger it now or in the future - narrower
#: and safer than reworking the scheduler's join/dispatch semantics to
#: encode an ordering the tool layer can just as well guarantee itself.
_cwd_locks: dict[str, asyncio.Lock] = {}


def _lock_for(cwd: Path) -> asyncio.Lock:
    key = str(cwd)
    lock = _cwd_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _cwd_locks[key] = lock
    return lock


#: How much combined stdout/stderr to fold into a failed `ToolOutcome.error`.
#: The tail is kept, not the head: `dotnet build`'s actual diagnostics (a
#: NuGet restore conflict, a compiler error) are printed just before its
#: final "Build FAILED" summary, however much restore/log noise came first.
_DIAGNOSTIC_TAIL_CHARS = 4000


async def default_runner(argv: Sequence[str], cwd: Path) -> tuple[int, str, str]:
    async with _lock_for(cwd):
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_SUBPROCESS_ENV,
        )
        stdout, stderr = await proc.communicate()
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


def _failure_message(argv: Sequence[str], returncode: int, stdout: str, stderr: str) -> str:
    """The generic `` `argv` exited N `` summary, plus the tool's own
    diagnostic - without this, a repair agent reading this string back via
    `ContextRetriever.last_error_of` sees only the exit code and never the
    actual compiler/NuGet/git error that caused it, making any fix a blind
    guess (see `agents/implementer.py`'s `repair*` positions, which read
    exactly this field as `<<<PRIOR_FAILURE>>>`)."""
    header = f"`{' '.join(argv)}` exited {returncode}"
    diagnostic = "\n".join(part.strip() for part in (stdout, stderr) if part.strip())
    if not diagnostic:
        return header
    if len(diagnostic) > _DIAGNOSTIC_TAIL_CHARS:
        diagnostic = "...(truncated)...\n" + diagnostic[-_DIAGNOSTIC_TAIL_CHARS:]
    return f"{header}:\n{diagnostic}"


async def run_argv(runner: SubprocessRunner, argv: Sequence[str], ctx: ToolContext) -> ToolOutcome:
    returncode, stdout, stderr = await runner(argv, Path(ctx.cwd))
    return ToolOutcome(
        ok=returncode == 0,
        output={"returncode": returncode, "stdout": stdout, "stderr": stderr},
        error=None if returncode == 0 else _failure_message(argv, returncode, stdout, stderr),
    )
