"""Filesystem tools: the only way anything in this system reads or writes a
file (CLAUDE.md section 6's "Repository" tool category).

Both handlers resolve their `path` argument strictly under `ctx.cwd` -
`..` traversal and absolute paths are rejected in the handler itself, before
the registry's own `writable_paths` check ever runs, since a path that has
already escaped the sandbox root is not something `ToolSpec.allows_write_to`
can meaningfully evaluate.

**Doubled-prefix guard, added after a live run's `_write_file` silently
wrote to the wrong place.** `agents/implementer.py`/`tester.py`/`docs.py`
ask the model for paths "relative to the solution root", meaning relative to
`ctx.cwd` (`.../src/url-shortener`) - but a repair attempt, fixing a real
`dotnet build` failure with a solution-wide `Directory.Build.targets`, wrote
it to `"src/url-shortener/Directory.Build.targets"`. That resolved to
`.../src/url-shortener/src/url-shortener/Directory.Build.targets` - one
level too deep for any project's MSBuild import search to ever find - and
`_write_file` reported `ok=True` regardless, since nothing about that path is
otherwise invalid. The fix was correct; it just never took effect, and the
same build failure repeated identically until the run's `cycle_budget` ran
out. `_write_file` now refuses (rather than silently mis-placing) a path
whose leading segments literally repeat `ctx.cwd`'s own trailing segments,
so this class of mistake surfaces immediately as a normal tool failure - fed
back to the next repair attempt exactly like a compiler error would be -
instead of being discovered later by a human reading the sandbox by hand.

**Project/solution-file guard, added after the same mistake recurred twice
in a row on a *different* file.** Two separate live runs had `impl_api`
"modify" `UrlShortener.Api.csproj` (to add a `ProjectReference` it believed
it needed) by regenerating the *entire* file from its own training-data
habits, silently reverting `<TargetFramework>` from the `net10.0`
`ScaffoldAgent` actually created (via `kernel/tools/dotnet.py`'s `-f
net10.0`) back to `net8.0` - a `NU1201` restore error identical both times,
burning `build_api`'s `cycle_budget` before either run got any further.
Project/solution structure is deliberately not a code-rewriting agent's to
own (`agents/scaffold.py`'s own docstring: "project *structure*, not file
*content*"), and it was never necessary in the first place - SDK-style
`ProjectReference`s are transitive, so `Api -> Infrastructure -> Application
-> Domain` (`ScaffoldAgent`'s own linear reference chain) already gives
`Api` compile-time access to every earlier project's public types with no
direct reference required. `_write_file` now refuses `.csproj`/`.sln`/
`.slnx` paths outright, for every caller - only `agents/scaffold.py`'s
`dotnet.*` tool calls create or modify those, and it never uses
`fs.write_file` to do so.

**Shared-file conflict guard, added after two concurrent repair agents
silently clobbered each other.** Live run `b5da55c3-...`: `build_domain` and
`build_api` failed with byte-for-byte identical `NU1510` diagnostics (both
were really the same whole-solution build - see `kernel/tools/dotnet.py`'s
now-fixed project scoping), so `repair_domain` and `repair_api` were
dispatched *together* by the scheduler's genuine `asyncio.gather`
concurrency (`kernel.scheduler.Scheduler._step`). Both independently decided
the fix was a solution-wide `Directory.Build.targets`, and both called
`fs.write_file` for that exact path in the same pass. `_write_file` was a
plain `write_text` with no read-before-write of any kind, so whichever call
physically executed second silently discarded the first repair's entire fix
- no error, no event, `node.succeeded` reported for both.

`_write_file` now does optimistic concurrency control for a small, named set
of shared MSBuild files (`Directory.Build.props`/`.targets` - the two
customization points `agents/implementer.py`'s prompt already tells the
model to use for a solution-wide fix): overwriting one that already exists
with genuinely different content requires an `expected_content` argument
that matches what is *actually* on disk right now, exactly like a
compare-and-swap. Writing identical content is always a no-op success
regardless (matching `WRITE_FILE.idempotent`'s existing claim), and creating
the file for the first time never needs `expected_content` at all. A plain
"does the new content differ from the old" check (tried first, and wrong) is
not the same thing: a *legitimate*, later, sequential edit to a file that
already has different content by design would look identical to a race to
that check and be refused forever, since a real fix is *always* different
content from what it is fixing. Compare-and-swap distinguishes them
correctly: a caller that actually read the file first and passes back what
it read is making an informed change (allowed, even though content differs);
a caller that never read it, or whose `expected_content` no longer matches
because someone else wrote in between, is not (refused, with the current
content returned so it can reconcile). `agents/implementer.py`'s
`ImplementerAgent` is the caller that does this: it reads both shared
filenames before ever asking the model to write, shows the model whatever
already exists so it can merge into it, and passes that same content back as
`expected_content`; a `SHARED_FILE_CONFLICT_PREFIX` refusal there triggers
one bounded reconciliation retry, re-reading and feeding the actual current
content back through the same `prior_error_section` mechanism already used
for a `dotnet build` failure. Ordinary per-file writes (every `.cs` file any
implementer/tester/docs call has ever made) are entirely unaffected: the
guard only ever evaluates the two named MSBuild filenames.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from ases.kernel.tools.classification import (
    SideEffect,
    ToolContext,
    ToolOutcome,
    ToolSpec,
    is_safe_relative_path,
)

#: Extensions no `fs.write_file` caller may ever target - see the module
#: docstring's project/solution-file guard note. Case-insensitive: Windows
#: paths and a model's own casing choices are not to be trusted to match.
_PROJECT_STRUCTURE_SUFFIXES = frozenset({".csproj", ".sln", ".slnx"})

#: Filenames (case-insensitive, wherever they appear) `fs.write_file` treats
#: as solution-wide and shared - see the module docstring's shared-file
#: conflict guard note. Deliberately just these two: they are MSBuild's own
#: fixed customization points (imported automatically into every project in
#: the directory tree), the only files more than one implementer/repair
#: agent has ever had a real reason to write in the same run.
_SHARED_MSBUILD_FILENAMES = frozenset({"directory.build.props", "directory.build.targets"})


class PathEscapesSandboxError(ValueError):
    """A requested path resolves outside `ToolContext.cwd`."""


class DoublesCwdPrefixError(ValueError):
    """A requested path's leading segment(s) literally repeat `ctx.cwd`'s own
    trailing segment(s) - see the module docstring's doubled-prefix note."""


def _doubles_cwd_prefix(cwd: PurePosixPath, requested: PurePosixPath) -> bool:
    cwd_parts, req_parts = cwd.parts, requested.parts
    depth = min(len(cwd_parts), len(req_parts))
    return any(req_parts[:k] == cwd_parts[-k:] for k in range(1, depth + 1))


def _resolve_under_cwd(ctx: ToolContext, raw_path: str) -> Path:
    if not is_safe_relative_path(raw_path):
        raise PathEscapesSandboxError(raw_path)
    requested = PurePosixPath(raw_path)
    if _doubles_cwd_prefix(PurePosixPath(ctx.cwd), requested):
        raise DoublesCwdPrefixError(raw_path)
    return Path(ctx.cwd) / Path(*requested.parts)


def _doubled_prefix_message(ctx: ToolContext, raw_path: str) -> str:
    return (
        f"refusing path {raw_path!r}: its leading segment(s) repeat the solution "
        f"root's own trailing path ({PurePosixPath(ctx.cwd).as_posix()!r}). Paths are "
        "already relative to the solution root - do not prefix them with it again."
    )


#: Prefix an `ImplementerAgent` reconciliation retry keys off of - see
#: `agents/implementer.py`'s `_CONFLICT_PREFIX`, which must match this
#: exactly. Kept as a plain string constant (not imported either direction,
#: `kernel/` must never import `agents/`) rather than a shared exception
#: type, since the only thing a caller needs from this is the message text.
SHARED_FILE_CONFLICT_PREFIX = "shared file conflict:"


def _shared_file_conflict_message(raw_path: str, current_content: str) -> str:
    return (
        f"{SHARED_FILE_CONFLICT_PREFIX} {raw_path!r} already has different content than "
        "what you are about to write, and either no expected_content was given or it no "
        "longer matches - it may have just been written by a different, concurrently "
        "running task fixing an unrelated problem. Its current content is:\n"
        f"<<<CURRENT_CONTENT>>>\n{current_content}\n<<<END_CURRENT_CONTENT>>>\n"
        "Regenerate this file merged with what is already there, not as a fresh replacement."
    )


def _project_structure_message(raw_path: str) -> str:
    return (
        f"refusing to write {raw_path!r}: project/solution files "
        f"({sorted(_PROJECT_STRUCTURE_SUFFIXES)}) are materialized and wired by "
        "ScaffoldAgent's dotnet.* tool calls only, never by rewriting the file. "
        "SDK-style ProjectReferences are transitive, so a needed type from an "
        "earlier project in the dependency chain is already accessible without "
        "adding a direct reference; write source/config files instead."
    )


async def _read_file(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
    raw_path = str(args["path"])
    try:
        target = _resolve_under_cwd(ctx, raw_path)
    except PathEscapesSandboxError as exc:
        return ToolOutcome(ok=False, error=f"path escapes sandbox: {exc}")
    except DoublesCwdPrefixError:
        return ToolOutcome(ok=False, error=_doubled_prefix_message(ctx, raw_path))
    if not target.is_file():
        return ToolOutcome(ok=False, error=f"no such file: {args['path']!r}")
    return ToolOutcome(ok=True, output={"content": target.read_text(encoding="utf-8")})


async def _write_file(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
    raw_path = str(args["path"])
    if PurePosixPath(raw_path).suffix.lower() in _PROJECT_STRUCTURE_SUFFIXES:
        return ToolOutcome(ok=False, error=_project_structure_message(raw_path))
    try:
        target = _resolve_under_cwd(ctx, raw_path)
    except PathEscapesSandboxError as exc:
        return ToolOutcome(ok=False, error=f"path escapes sandbox: {exc}")
    except DoublesCwdPrefixError:
        return ToolOutcome(ok=False, error=_doubled_prefix_message(ctx, raw_path))
    content = str(args["content"])
    if target.name.lower() in _SHARED_MSBUILD_FILENAMES and target.is_file():
        current = target.read_text(encoding="utf-8")
        if current != content:
            expected = args.get("expected_content")
            if expected is None or str(expected) != current:
                return ToolOutcome(ok=False, error=_shared_file_conflict_message(raw_path, current))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return ToolOutcome(ok=True, output={"bytes_written": target.stat().st_size})


READ_FILE = ToolSpec(
    name="fs.read_file",
    handler=_read_file,
    idempotent=True,
    side_effect=SideEffect.NONE,
)

WRITE_FILE = ToolSpec(
    name="fs.write_file",
    handler=_write_file,
    idempotent=True,  # writing identical content twice is a no-op in effect
    side_effect=SideEffect.SANDBOX,
    writable_paths=(PurePosixPath("."),),
    path_args=("path",),
)
