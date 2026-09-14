"""Structural validation of a scaffolded solution, before anything is built.

Three checks run before a single `dotnet build` is dispatched. The first
lives in `agents.planner.TaskGraphSubgraphProvider._check_plan_fits_the_solution`
and is about the *plan*: every task names a real project, and every project
the architecture called for has someone implementing it. This module holds
the other two: `check_reference_graph` is about the *solution on disk* - the
project reference graph the scaffold actually produced - and
`check_contract_graph` is about the *plan's contract linkage* - whether
`TaskSpec.produces_contracts`/`consumes_contracts` actually forms a
consistent graph against `SolutionSkeleton.frozen_interfaces`, so a
consuming task's context (`agents/implementer.py`) can be built from a
contract the graph has already verified exists, is uniquely owned, and is
reachable - rather than trusting an unverified decomposer claim.

**Why this is worth doing separately from the compiler.** `dotnet build` will
of course notice a missing type eventually - but by then the run has spent an
implementation call per task, a build per project, and however much of each
task's repair budget the failure consumes, and the diagnostic it produces
names the consuming project rather than the layer that is actually missing
something. Live run `009ea59f-...` spent three builds and two repair attempts
before safe-stopping with a halt reason that named `build_api`'s exhausted
cycle budget - while the real fault was an unimplemented Application layer
several edges upstream. A reference cycle or an inverted dependency is
cheaper to find by reading the `.csproj` files.

**What this deliberately does not do.** It reads project *references*, not
source. Assertions about what the code inside a project imports - that a
domain layer touches no ORM, that an application layer does not reach into
infrastructure - need a `using`-level index of the generated C#, which is
what `ases/codebase/` is reserved for and does not yet contain. Claiming
those checks here by pattern-matching file text would be worse than not
making them: it would report confident architectural verdicts from evidence
that cannot support them. The reference-direction check below is real,
deterministic, and honest about its scope; the generated `*.ArchitectureTests`
project is where the source-level rules belong, as real tests inside the
solution.

Generic by construction: nothing here knows what a "domain" or an "API" is.
It is handed the declared layer order (`DesignSpec.layers` as materialized
into `SolutionSkeleton.projects`, which the scaffold agent is already
instructed to emit in dependency order) and checks the references against
that, whatever the layers happen to be called.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from ases.contracts.artifacts import SolutionSkeleton, TaskGraph

#: `<ProjectReference Include="..\Other\Other.csproj" />`, tolerant of
#: attribute order, quoting style and whitespace. Deliberately a regex over
#: the raw file rather than an XML parse: a half-written `.csproj` should
#: produce "no references found", not a parse exception that fails a run for
#: a reason unrelated to what is being checked.
_PROJECT_REFERENCE = re.compile(
    r"<ProjectReference\b[^>]*\bInclude\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE
)


class StructuralFinding(BaseModel):
    """One structural problem, with enough detail to act on."""

    model_config = ConfigDict(frozen=True)

    rule: str
    message: str
    #: The project the finding is about, when it is about one.
    project: str | None = None


class StructureReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    findings: tuple[StructuralFinding, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.findings

    def summary(self) -> str:
        if self.ok:
            return "no structural problems found"
        return "; ".join(f"[{f.rule}] {f.message}" for f in self.findings)


def project_name_of(reference: str) -> str:
    """The project name a `ProjectReference` Include path points at.

    `..\\UrlShortener.Domain\\UrlShortener.Domain.csproj` -> `UrlShortener.Domain`.
    Handles both separators, because a `.csproj` written on Windows and one
    written by `dotnet` on Linux differ here and neither is wrong.
    """
    tail = reference.replace("\\", "/").rsplit("/", 1)[-1]
    return tail[: -len(".csproj")] if tail.lower().endswith(".csproj") else tail


def references_in(csproj_text: str) -> tuple[str, ...]:
    """Project names referenced by one `.csproj`'s text, in file order."""
    return tuple(project_name_of(m) for m in _PROJECT_REFERENCE.findall(csproj_text))


def check_reference_graph(
    *,
    projects: Sequence[str],
    csproj_by_project: Mapping[str, str],
) -> StructureReport:
    """Check the reference graph the scaffold produced against the declared
    layer order.

    `projects` is `SolutionSkeleton.projects`, which the scaffold agent is
    instructed to emit in dependency order - a project later in the list may
    depend on an earlier one, never the reverse. That ordering is the only
    architectural input, and it came from the architecture agent by way of the
    scaffold: this function has no opinion of its own about what the layers
    should be.

    Three rules, all deterministic:

    - `unknown_reference`: a project references something not in the solution.
    - `inverted_dependency`: a project references one declared *after* it,
      which is the layering violation the ordering exists to prevent - an
      application layer referencing infrastructure, for instance.
    - `reference_cycle`: the reference graph is not a DAG. MSBuild rejects
      this too, but with an error that names an arbitrary member of the cycle
      rather than the cycle.

    A project with no `.csproj` text supplied is skipped, not reported: this
    runs against whatever the sandbox actually contains, and "I could not read
    that file" is not a structural finding about the architecture.
    """
    declared = list(projects)
    position = {name: index for index, name in enumerate(declared)}
    findings: list[StructuralFinding] = []
    edges: dict[str, list[str]] = {}

    for project in declared:
        text = csproj_by_project.get(project)
        if text is None:
            continue
        referenced = references_in(text)
        edges[project] = [r for r in referenced if r in position]
        for reference in referenced:
            if reference not in position:
                findings.append(
                    StructuralFinding(
                        rule="unknown_reference",
                        project=project,
                        message=(
                            f"{project} references {reference!r}, which is not one of the "
                            f"solution's projects: {declared}"
                        ),
                    )
                )
            elif position[reference] > position[project]:
                findings.append(
                    StructuralFinding(
                        rule="inverted_dependency",
                        project=project,
                        message=(
                            f"{project} references {reference}, which the architecture "
                            f"declares after it. Dependencies run one way: a project may "
                            f"reference one declared earlier, never one declared later"
                        ),
                    )
                )

    if cycle := _first_cycle(edges):
        findings.append(
            StructuralFinding(
                rule="reference_cycle",
                project=cycle[0],
                message=f"project references form a cycle: {' -> '.join(cycle)}",
            )
        )

    return StructureReport(findings=tuple(findings))


def check_contract_graph(*, task_graph: TaskGraph, skeleton: SolutionSkeleton) -> StructureReport:
    """Check `TaskSpec.produces_contracts`/`consumes_contracts` against
    `SolutionSkeleton.frozen_interfaces`, deterministically - no C# source is
    read here, only the two structured artifacts the plan and the scaffold
    already produced.

    Five rules:

    - `unknown_produced_contract` / `unknown_consumed_contract`: a task names
      a `type_name` `skeleton.frozen_interfaces` does not declare.
    - `contract_project_mismatch`: a task claims to produce an interface
      declared (by `FrozenInterface.project`) in a different project than
      the task's own `component` - the task cannot be the one writing it.
    - `duplicate_contract_producer`: two tasks both claim to produce the same
      `type_name` - ambiguous ownership, and exactly the shape that let two
      tasks independently guess at (and diverge on) one interface before
      `FrozenInterface` existed.
    - `contract_without_producer`: a task consumes a `type_name` no task in
      the graph produces - the direct check for "every consumed contract has
      a producer."
    - `contract_reference_missing`: a consuming task's project cannot reach
      the producing interface's project. `skeleton.projects` is scaffolded as
      a strictly linear reference chain (`agents/scaffold.py`), so "reachable"
      is exactly `check_reference_graph`'s own `position` ordering: a project
      may use an earlier project's types, never a later one's.
    """
    findings: list[StructuralFinding] = []
    interfaces_by_name = {i.type_name: i for i in skeleton.frozen_interfaces}
    position = {name: index for index, name in enumerate(skeleton.projects)}

    producers: dict[str, list[str]] = {}
    for task in task_graph.tasks:
        for name in task.produces_contracts:
            producers.setdefault(name, []).append(task.id)
            interface = interfaces_by_name.get(name)
            if interface is None:
                findings.append(
                    StructuralFinding(
                        rule="unknown_produced_contract",
                        project=task.component,
                        message=(
                            f"task {task.id!r} claims to produce {name!r}, which is not "
                            "declared in the scaffold's frozen_interfaces"
                        ),
                    )
                )
            elif interface.project != task.component:
                findings.append(
                    StructuralFinding(
                        rule="contract_project_mismatch",
                        project=task.component,
                        message=(
                            f"task {task.id!r} (component {task.component!r}) claims to "
                            f"produce {name!r}, which the scaffold declares in project "
                            f"{interface.project!r} instead"
                        ),
                    )
                )

    for name, task_ids in producers.items():
        if len(task_ids) > 1:
            findings.append(
                StructuralFinding(
                    rule="duplicate_contract_producer",
                    message=f"{name!r} is produced by more than one task: {sorted(task_ids)}",
                )
            )

    for task in task_graph.tasks:
        for name in task.consumes_contracts:
            interface = interfaces_by_name.get(name)
            if interface is None:
                findings.append(
                    StructuralFinding(
                        rule="unknown_consumed_contract",
                        project=task.component,
                        message=(
                            f"task {task.id!r} consumes {name!r}, which is not declared in "
                            "the scaffold's frozen_interfaces"
                        ),
                    )
                )
                continue
            if name not in producers:
                findings.append(
                    StructuralFinding(
                        rule="contract_without_producer",
                        project=task.component,
                        message=(
                            f"task {task.id!r} consumes {name!r}, which no task in the plan "
                            "produces"
                        ),
                    )
                )
                continue
            producer_position = position.get(interface.project)
            consumer_position = position.get(task.component)
            if (
                producer_position is not None
                and consumer_position is not None
                and producer_position > consumer_position
            ):
                findings.append(
                    StructuralFinding(
                        rule="contract_reference_missing",
                        project=task.component,
                        message=(
                            f"task {task.id!r} (component {task.component!r}) consumes "
                            f"{name!r}, declared in project {interface.project!r}, which the "
                            "architecture declares after it - a project may only reference "
                            "one declared earlier, never one declared later"
                        ),
                    )
                )

    return StructureReport(findings=tuple(findings))


def _first_cycle(edges: Mapping[str, Sequence[str]]) -> list[str] | None:
    """A reference cycle as a readable path, or None."""
    white, grey, black = 0, 1, 2
    colour = dict.fromkeys(edges, white)
    for root in edges:
        if colour[root] != white:
            continue
        stack: list[tuple[str, int]] = [(root, 0)]
        path: list[str] = []
        while stack:
            node, index = stack[-1]
            if index == 0:
                colour[node] = grey
                path.append(node)
            children = edges.get(node, ())
            if index < len(children):
                stack[-1] = (node, index + 1)
                child = children[index]
                if colour.get(child) == grey:
                    return [*path[path.index(child) :], child]
                if colour.get(child, black) == white:
                    stack.append((child, 0))
            else:
                colour[node] = black
                path.pop()
                stack.pop()
    return None


__all__ = [
    "StructuralFinding",
    "StructureReport",
    "check_contract_graph",
    "check_reference_graph",
    "project_name_of",
    "references_in",
]
