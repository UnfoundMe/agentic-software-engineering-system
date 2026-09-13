"""Filesystem tools: the only way anything in this system reads or writes a
file (CLAUDE.md section 6's "Repository" tool category).

Both handlers resolve their `path` argument strictly under `ctx.cwd` -
`..` traversal and absolute paths are rejected in the handler itself, before
the registry's own `writable_paths` check ever runs, since a path that has
already escaped the sandbox root is not something `ToolSpec.allows_write_to`
can meaningfully evaluate.
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


class PathEscapesSandboxError(ValueError):
    """A requested path resolves outside `ToolContext.cwd`."""


def _resolve_under_cwd(ctx: ToolContext, raw_path: str) -> Path:
    if not is_safe_relative_path(raw_path):
        raise PathEscapesSandboxError(raw_path)
    requested = PurePosixPath(raw_path)
    return Path(ctx.cwd) / Path(*requested.parts)


async def _read_file(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
    try:
        target = _resolve_under_cwd(ctx, str(args["path"]))
    except PathEscapesSandboxError as exc:
        return ToolOutcome(ok=False, error=f"path escapes sandbox: {exc}")
    if not target.is_file():
        return ToolOutcome(ok=False, error=f"no such file: {args['path']!r}")
    return ToolOutcome(ok=True, output={"content": target.read_text(encoding="utf-8")})


async def _write_file(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
    try:
        target = _resolve_under_cwd(ctx, str(args["path"]))
    except PathEscapesSandboxError as exc:
        return ToolOutcome(ok=False, error=f"path escapes sandbox: {exc}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(args["content"]), encoding="utf-8")
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
