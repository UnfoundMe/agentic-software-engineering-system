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

from ases.contracts.artifacts import SolutionSkeleton, TaskGraph
from ases.kernel.tools.classification import SideEffect, ToolContext, ToolOutcome, ToolSpec
from ases.validation.structure import StructureReport, check_contract_graph, check_reference_graph


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


def _contract_report(args: Mapping[str, Any]) -> StructureReport:
    """`check_contract_graph` over `tasks`/`frozen_interfaces`, when both are
    given - purely additive: a caller with no task graph yet (or predating
    this check) omits them and gets an empty report, same as before."""
    raw_tasks = args.get("tasks")
    raw_interfaces = args.get("frozen_interfaces")
    if not raw_tasks or raw_interfaces is None:
        return StructureReport()
    task_graph = TaskGraph.model_validate({"tasks": raw_tasks})
    skeleton = SolutionSkeleton.model_validate(
        {"projects": args.get("projects", ()), "frozen_interfaces": raw_interfaces}
    )
    return check_contract_graph(task_graph=task_graph, skeleton=skeleton)


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
    reference_report = check_reference_graph(projects=projects, csproj_by_project=texts)
    contract_report = _contract_report(args)
    report = StructureReport(findings=(*reference_report.findings, *contract_report.findings))
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
