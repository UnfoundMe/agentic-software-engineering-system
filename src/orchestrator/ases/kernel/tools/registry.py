"""The deny-by-default tool registry (docs/05 section 3.5, CLAUDE.md section 6).

Three refusals are enforced *here*, structurally, before a handler ever runs -
not left to each handler to remember:

1. An unregistered tool name.
2. A write path outside the tool's declared `writable_paths`
   (`ToolSpec.path_args` names which arguments are paths).
3. A call that exceeds the tool's declared `timeout_s`.

`invoke` never raises for any of these; it returns a `ToolInvocationResult`
whose `verdict` says exactly what happened. This mirrors `kernel.gates`'s
`BudgetEntryGate` (a pure evaluator returning `GateResult`, not an exception) -
the registry has no `EventStore` and emits nothing itself; the caller (a
future `NodeExecutor`, or a test) turns a denial into a `POLICY_VIOLATION` or
`TOOL_FAILED` event, exactly as the scheduler already does for gate results.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from ases.kernel.tools.classification import (
    ToolContext,
    ToolOutcome,
    ToolSpec,
    is_safe_relative_path,
)


class InvocationVerdict(StrEnum):
    EXECUTED = "executed"  # the handler ran; see `ok` for whether it succeeded
    DENIED_UNKNOWN_TOOL = "denied_unknown_tool"
    DENIED_WRITE_OUTSIDE_SANDBOX = "denied_write_outside_sandbox"
    DENIED_TIMED_OUT = "denied_timed_out"


@dataclass(frozen=True)
class ToolInvocationResult:
    verdict: InvocationVerdict
    ok: bool
    output: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def denied(self) -> bool:
        return self.verdict is not InvocationVerdict.EXECUTED


class UnknownToolNameError(KeyError):
    """`ToolRegistry.register` guards against a duplicate name; this is
    raised only by `spec_for`, for a caller that wants the `ToolSpec` itself
    rather than an invocation result."""


class ToolRegistry:
    """Holds every registered `ToolSpec`, keyed by name. Deny-by-default:
    a name never registered has no path to execution at all."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} is already registered")
        self._tools[spec.name] = spec

    def spec_for(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownToolNameError(name) from None

    def known_tools(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    async def invoke(
        self, name: str, args: Mapping[str, Any], ctx: ToolContext
    ) -> ToolInvocationResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolInvocationResult(
                verdict=InvocationVerdict.DENIED_UNKNOWN_TOOL,
                ok=False,
                error=f"no tool registered under name {name!r}",
            )

        violation = self._writable_path_violation(spec, args)
        if violation is not None:
            return ToolInvocationResult(
                verdict=InvocationVerdict.DENIED_WRITE_OUTSIDE_SANDBOX,
                ok=False,
                error=(
                    f"tool {name!r} attempted to write to {violation}, which is outside "
                    f"its declared writable_paths {spec.writable_paths}"
                ),
            )

        try:
            outcome = await asyncio.wait_for(spec.handler(args, ctx), timeout=spec.timeout_s)
        except TimeoutError:
            return ToolInvocationResult(
                verdict=InvocationVerdict.DENIED_TIMED_OUT,
                ok=False,
                error=f"tool {name!r} exceeded its {spec.timeout_s}s timeout",
            )

        return ToolInvocationResult(
            verdict=InvocationVerdict.EXECUTED,
            ok=outcome.ok,
            output=outcome.output,
            error=outcome.error,
        )

    @staticmethod
    def _writable_path_violation(spec: ToolSpec, args: Mapping[str, Any]) -> PurePosixPath | None:
        for key in spec.path_args:
            value = args.get(key)
            if value is None:
                continue
            raw = str(value)
            # `is_safe_relative_path` rejects `..` traversal, a leading `/`,
            # and (found while testing on Windows) a `\`- or `:`-bearing
            # string that would otherwise be treated as a single opaque path
            # segment and later resolved as a real Windows-absolute path,
            # silently discarding the sandbox root - see its docstring.
            if not is_safe_relative_path(raw):
                return PurePosixPath(raw.replace("\\", "/"))
            path = PurePosixPath(raw)
            if not spec.allows_write_to(path):
                return path
        return None


__all__ = [
    "InvocationVerdict",
    "ToolContext",
    "ToolInvocationResult",
    "ToolOutcome",
    "ToolRegistry",
    "ToolSpec",
    "UnknownToolNameError",
]
