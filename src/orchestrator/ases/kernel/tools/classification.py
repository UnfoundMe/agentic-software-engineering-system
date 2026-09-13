"""What a tool is allowed to do, declared once at registration time.

This is the metadata `kernel.recovery` (Phase 5) will consult to decide retry
and compensation strategy, and what `ToolRegistry` (this package) consults
right now to enforce writable-path and approval boundaries before a tool ever
runs. Every registered tool declares exactly this shape - docs/05 section 3.5.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any


class SideEffect(StrEnum):
    """Selects the recovery mechanism (docs/05 section 5), not just a label."""

    NONE = "none"  # pure; freely retryable
    SANDBOX = "sandbox"  # confined to the worktree; rollback is free and total
    EXTERNAL = "external"  # escapes the sandbox; needs an ordered compensator


ToolHandler = Callable[[Mapping[str, Any], "ToolContext"], Awaitable["ToolOutcome"]]


@dataclass(frozen=True)
class ToolContext:
    """The one piece of caller-supplied state a tool handler receives.

    `cwd` is the sandbox root a tool may operate under - never the real
    repository (docs/05 section 3.5, CLAUDE.md section 3). Plain `Path`,
    never a `sandbox.Workspace`: kernel modules take primitive values instead
    of importing the layer above them, exactly as `BudgetLimits` mirrors
    `Settings` fields without importing `config` (kernel/gates.py).
    """

    cwd: PurePosixPath
    run_id: str
    node_id: str | None = None


@dataclass(frozen=True)
class ToolOutcome:
    """What a tool handler itself reports - distinct from `ToolInvocationResult`,
    which additionally records whether the registry ever let the handler run
    at all."""

    ok: bool
    output: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    """Declared once per tool, consulted by the registry before every call.

    `writable_paths` is empty for a read-only tool; a tool that writes
    declares the path prefixes (relative to `ToolContext.cwd`) it may write
    under, and the registry - not the handler - is what enforces that a
    requested write path falls under one of them.
    """

    name: str
    handler: ToolHandler
    idempotent: bool
    side_effect: SideEffect
    requires_approval: bool = False
    compensator: str | None = None
    timeout_s: float = 60.0
    writable_paths: tuple[PurePosixPath, ...] = ()
    #: Argument keys (in the `args` mapping passed to `invoke`) whose value is
    #: a path the tool will write to. The registry resolves each named
    #: argument to a path and checks it against `writable_paths` *before*
    #: the handler ever runs - this is what makes "write outside
    #: writable_paths -> DENY" a property of the registry, not something each
    #: handler must remember to re-check itself.
    path_args: tuple[str, ...] = ()

    def allows_write_to(self, path: PurePosixPath) -> bool:
        return any(path.is_relative_to(prefix) for prefix in self.writable_paths)


#: Characters that make a string unsafe to treat as a POSIX-relative path
#: component, found while testing this module on Windows rather than assumed:
#: `PurePosixPath` never splits on `\`, so a Windows-style path
#: (`C:\Users\x`) becomes a single opaque path *segment*. When that segment
#: is later passed to `pathlib.Path(*parts)` on a Windows host, the
#: single-string form is recognised as a full absolute path by the platform
#: parser, and joining it onto a sandbox root silently discards the root
#: entirely - confirmed interactively before this check was added. A colon
#: is rejected too, since POSIX relative paths never need one and Windows
#: reserves it for drive letters.
_UNSAFE_PATH_CHARS = ("\\", ":")


def is_safe_relative_path(raw: str) -> bool:
    """True only if `raw` cannot escape a sandbox root when resolved against
    it: no POSIX-absolute leading slash, no `..` traversal, and none of
    `_UNSAFE_PATH_CHARS`. Every `path_args` value the registry checks, and
    every path a filesystem tool handler resolves, must pass this first."""
    if any(ch in raw for ch in _UNSAFE_PATH_CHARS):
        return False
    path = PurePosixPath(raw)
    return not path.is_absolute() and ".." not in path.parts
