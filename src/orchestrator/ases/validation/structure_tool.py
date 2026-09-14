"""The `structure.check_solution` tool: read-only structural validation of a
scaffolded solution, before anything is compiled.

Lives in `validation/` rather than `kernel/tools/` deliberately. A `ToolSpec`
is data - the registry that enforces it is kernel-side, the *definition* need
not be - and `tests/invariants/test_layering.py` forbids `kernel/` from
importing `validation`. Putting the spec next to the check it wraps keeps the
dependency pointing the right way: validation may reach down into the kernel's
tool vocabulary, never the reverse.

Read-only by construction: `SideEffect.NONE`, no `writable_paths`,
idempotent. It reads `.csproj` files under the sandbox and nothing else.

A tool rather than a helper some agent calls, because the check is
deterministic, its verdict is authoritative (CLAUDE.md section 10), and it
must be able to fail a node. A layering fault the compiler would surface
several LLM calls later - as a `CS0246` in whichever project happens to
consume the missing type - should stop the run here, named for what it
actually is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ases.kernel.tools.classification import SideEffect, ToolContext, ToolOutcome, ToolSpec
from ases.validation.structure import check_reference_graph


def _csproj_text(root: Path, project: str) -> str | None:
    """`<root>/<project>/<project>.csproj` - the layout `dotnet new -o <name>`
    produces and `agents/scaffold.py` relies on.

    None when absent, so a project that was never materialized is skipped
    rather than reported: "not there" is a scaffolding failure, and the
    scaffold step already reports its own.
    """
    candidate = root / project / f"{project}.csproj"
    if not candidate.is_file():
        return None
    try:
        return candidate.read_text(encoding="utf-8")
    except OSError:
        return None


async def _check_solution(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
    projects: Sequence[str] = [str(p) for p in args.get("projects", ())]
    if not projects:
        return ToolOutcome(
            ok=True,
            output={
                "checked": 0,
                "findings": [],
                "summary": "no projects declared; nothing to check",
            },
        )

    root = Path(str(ctx.cwd))
    texts = {p: text for p in projects if (text := _csproj_text(root, p)) is not None}
    report = check_reference_graph(projects=projects, csproj_by_project=texts)
    return ToolOutcome(
        ok=report.ok,
        output={
            "checked": len(texts),
            "findings": [f.model_dump(mode="json") for f in report.findings],
            "summary": report.summary(),
        },
        error=None if report.ok else report.summary(),
    )


CHECK_SOLUTION = ToolSpec(
    name="structure.check_solution",
    handler=_check_solution,
    idempotent=True,
    side_effect=SideEffect.NONE,
)
