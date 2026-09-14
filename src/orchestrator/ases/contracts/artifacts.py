"""The artifact contracts, one per SDLC stage.

Grouped by lifecycle stage to match docs/03 section 3.3's node choreography
and `workflows/greenfield.yaml`'s `produces:` fields - each model here is one
of those. The `CONTRACTS` registry at the bottom is what
`tests/unit/test_contracts.py` uses to prove every kind a workflow YAML names
actually has a model, so the two cannot silently drift apart.

Field sets are deliberately modest. These are the first draft of a stable
interface for agents that do not exist yet (Phase 4); getting every field
right before a single agent has been built would be guessing. What matters
now is that every kind named in the workflow graph has *a* contract, and that
L1 schema validation (docs/05 section 6) has something real to check against.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime

from pydantic import Field, model_validator

from ases.contracts.base import ArtifactModel

# --- requirements -----------------------------------------------------


class Ambiguity(ArtifactModel):
    question: str
    resolution: str | None = None
    rationale: str | None = None
    resolved_by: str | None = None
    resolved_at: datetime | None = None

    @property
    def is_resolved(self) -> bool:
        return self.resolution is not None


class RequirementSpec(ArtifactModel):
    """Output of the requirements agent: the normalized engineering problem."""

    summary: str
    in_scope: tuple[str, ...] = ()
    out_of_scope: tuple[str, ...] = ()
    source_text: str


class AmbiguityRegister(ArtifactModel):
    """Every ambiguity the requirements agent surfaced, resolved or not.

    A human gate (docs/02 section 4.4) reviews this alongside `RequirementSpec`
    before the run may proceed - an unresolved ambiguity here is what Gate 1
    is checking for.
    """

    ambiguities: tuple[Ambiguity, ...] = ()

    @property
    def all_resolved(self) -> bool:
        return all(a.is_resolved for a in self.ambiguities)


# --- architecture (brownfield impact is Stage 2; the model exists now) -


class ImpactReport(ArtifactModel):
    """Stage 2 (Roslyn) output. Modelled now so the contract is stable before
    the analyzer exists - see docs/02 Phase 6."""

    impacted_modules: tuple[str, ...] = ()
    impacted_endpoints: tuple[str, ...] = ()
    impacted_tables: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class ArchitectureDecision(ArtifactModel):
    """One ADR. `DesignSpec` and its ADRs are produced by the same node
    (docs/03 section 3.3: 'DesignSpec + ADR-001') but are distinct artifacts,
    hashed and versioned independently."""

    title: str
    status: str = "proposed"
    context: str
    decision: str
    consequences: str = ""


class DesignSpec(ArtifactModel):
    """Output of the architect agent. Must state how the chosen structure
    satisfies claude.md section 9's layer separation (docs/03 section 3.1a) -
    that is guidance given to the agent, not a field this model enforces."""

    summary: str
    layers: tuple[str, ...] = ()
    key_decisions: tuple[str, ...] = ()


class ProjectPackages(ArtifactModel):
    """Which NuGet packages one specific project needs.

    Exists because the alternative - installing every `pinned_packages` entry
    into every project, which `agents/scaffold.py` did until a live run
    (`009ea59f-...`) showed what it costs - puts EF Core, Redis and xunit into
    a Domain project CLAUDE.md section 9 requires to depend on none of them,
    and puts framework-supplied packages into the web host, where .NET 10's
    package pruning rejects them outright (`NU1510` under `-warnaserror`).

    Deliberately declared by the scaffold agent rather than inferred from
    project names by the orchestrator: which layer needs which package is an
    architecture fact the agent that chose the architecture knows, and name
    inference is exactly what this system is not supposed to route on.
    """

    project: str
    packages: tuple[str, ...] = ()


class FrozenInterface(ArtifactModel):
    """One type whose shape *and location* parallel implementation tasks must
    agree on.

    This carried only `signature` until live run `91229361-...`. That run's
    Application task declared `IShortLinkCache` in `namespace
    UrlShortener.Application`; the Infrastructure cache task, running later
    and having never seen that code, wrote `using
    UrlShortener.Application.Ports;` and failed to compile against a type it
    had the exact signature of. A signature with no namespace pins what a
    type looks like and leaves where it lives to be guessed independently by
    every task that touches it - and a guess made by the declaring task and a
    guess made by the consuming task agree only by luck.

    `namespace` is what the compiler actually resolves against, so it is
    required. `project` says which assembly declares it, which is what tells
    a consumer whether it needs a `ProjectReference` at all.

    `type_name` and `file_path` were added for the same reason, one layer up:
    `TaskSpec.produces_contracts`/`consumes_contracts` (below) link a task to
    the interfaces it owns or needs by name, and a repair prompt needs to
    name the file a failure actually involves without parsing compiler
    output - both need a value distinct from the free-text `signature`.
    """

    #: The C# signature, e.g. `public interface IShortLinkCache { ... }`.
    signature: str
    #: The namespace it is declared in, verbatim - `UrlShortener.Application`,
    #: not `UrlShortener.Application.Abstractions` unless that is literally
    #: what the file says. The declaring task writes this namespace and every
    #: consuming task imports it; neither gets a choice.
    namespace: str = Field(min_length=1)
    #: The project whose assembly contains it.
    project: str = Field(min_length=1)
    #: The bare type/interface name alone, e.g. `IShortLinkCache` - no
    #: namespace prefix. What `TaskSpec.produces_contracts`/`consumes_contracts`
    #: name to link a task to this interface, since matching on the full
    #: `signature` string would be fragile (whitespace, modifiers, generic
    #: parameters all vary text that names the same type).
    type_name: str = Field(min_length=1)
    #: Where the type is expected to live, e.g.
    #: `UrlShortener.Application/IShortLinkCache.cs` - relative to the
    #: solution root, the same convention `agents/implementer.py`'s prompt
    #: already uses for a `FileChange.path`.
    file_path: str = Field(min_length=1)


class SolutionSkeleton(ArtifactModel):
    """Output of `SCAFFOLD` (docs/02 section 7.1): the frozen project
    structure and interface signatures that parallel implementation tasks
    build against - this is what removes interface drift as a failure mode."""

    projects: tuple[str, ...] = ()
    #: Every package the solution needs, across all projects. Retained as the
    #: solution-wide inventory (and as the fallback `scaffold` installs
    #: everywhere when `project_packages` is empty - see
    #: `ScaffoldAgent._packages_for`).
    pinned_packages: tuple[str, ...] = ()
    #: Per-project routing for `pinned_packages`. Empty means "no routing
    #: stated"; `scaffold` then falls back to the solution-wide behaviour.
    project_packages: tuple[ProjectPackages, ...] = ()
    #: The shared contract every parallel implementation task builds
    #: against: signature *and* namespace, so no task has to guess where a
    #: type its dependency declared actually lives. See `FrozenInterface`.
    frozen_interfaces: tuple[FrozenInterface, ...] = ()

    @model_validator(mode="after")
    def _type_names_are_unique(self) -> SolutionSkeleton:
        names = [i.type_name for i in self.frozen_interfaces]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate frozen interface type_name(s): {duplicates}")
        return self

    def frozen_interface_block(self, type_names: Collection[str] | None = None) -> str:
        """The frozen contract as a prompt-ready block, grouped by namespace.

        Rendering lives here rather than in each agent because `decompose`
        and `implementer` must show the model the *same* contract - one of
        them quietly dropping the namespace is the failure this type exists
        to prevent.

        `type_names`, when given, renders only the interfaces named -
        `agents/implementer.py` uses this to show a task the contracts it
        actually produces/consumes (`TaskSpec.produces_contracts`/
        `consumes_contracts`) instead of every frozen interface in the
        solution, which would only grow the prompt as a solution gets more
        tasks without making any one task's contract any clearer. `None`
        (the default) renders everything, unchanged from before this
        parameter existed - `agents/decompose.py` still wants the full
        landscape to assign tasks against.
        """
        interfaces = self.frozen_interfaces
        if type_names is not None:
            wanted = set(type_names)
            interfaces = tuple(i for i in interfaces if i.type_name in wanted)
        if not interfaces:
            return "(none stated)"
        by_namespace: dict[tuple[str, str], list[str]] = {}
        for interface in interfaces:
            by_namespace.setdefault((interface.namespace, interface.project), []).append(
                interface.signature
            )
        blocks = []
        for (namespace, project), signatures in by_namespace.items():
            body = chr(10).join(f"    {s}" for s in signatures)
            header = f"namespace {namespace};   // declared in project {project}"
            blocks.append(header + chr(10) + body)
        return (chr(10) * 2).join(blocks)


# --- planning -----------------------------------------------------------


class TaskSpec(ArtifactModel):
    """One unit of implementation work the decomposer proposes.

    `component` and `depends_on` are what make this executable rather than
    merely descriptive. Until they were consumed, `workflows/greenfield.yaml`
    carried a hard-coded `impl_domain`/`impl_api`/`impl_infrastructure`
    fan-out as a stand-in, and live run `009ea59f-...` showed the cost: the
    decomposer emitted five `app-*` tasks for a project the static graph had
    no node for, so `UrlShortener.Application` was never implemented at all
    and everything that compiles against it failed. The task list was right;
    nothing executed it.
    """

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    description: str
    #: Which project/component this task writes into. Must name one of
    #: `SolutionSkeleton.projects` - checked at subgraph admission
    #: (`agents.planner`), not here, since this model has no access to the
    #: skeleton. Empty means "not stated", which admission rejects.
    component: str = ""
    #: The task's role. Only `implementation` is executable today; the field
    #: exists so a decomposer can label work the runtime does not yet route
    #: (documentation, migration, benchmark) without that being indistinguishable
    #: from an implementation task it silently failed to run.
    kind: str = "implementation"
    #: Ids of tasks that must have been implemented **and compiled** before
    #: this one starts. Explicit, never inferred from naming, folder layout
    #: or list order - see `TaskGraph`'s validator.
    depends_on: tuple[str, ...] = ()
    #: File paths this task is expected to create or modify, relative to the
    #: solution root (the same convention `agents/implementer.py`'s prompt
    #: uses for a `FileChange.path`). Advisory, not enforced against what the
    #: task actually writes - it exists so a decomposer states a task's scope
    #: concretely enough to notice when one task is quietly covering several
    #: independent files/concerns, and so a repair prompt can name the files
    #: a component's failure involves without parsing compiler output.
    files: tuple[str, ...] = ()
    #: `FrozenInterface.type_name` values this task is responsible for
    #: declaring. Links a task to `SolutionSkeleton.frozen_interfaces` from
    #: the decompose side, since no `TaskSpec` exists yet when scaffold
    #: populates that list - see `FrozenInterface`'s own docstring.
    produces_contracts: tuple[str, ...] = ()
    #: `FrozenInterface.type_name` values this task consumes - normally
    #: produced by one of `depends_on`. `validation.structure.check_contract_graph`
    #: checks every consumed contract actually has a producer somewhere in
    #: the graph.
    consumes_contracts: tuple[str, ...] = ()


class TaskGraph(ArtifactModel):
    """Output of the decomposer, and the authority on what implementation
    work a run performs: `agents.planner.TaskGraphSubgraphProvider` turns it
    into a live subgraph admitted through `WorkflowGraph.with_subgraph`.

    The validator below runs at the LLM boundary, so a decomposer that
    proposes a self-contradictory plan produces an ordinary schema error -
    repaired once by `providers.structured.complete_structured`, and
    classified `AGENT_PROTOCOL_FAILURE` if that repair fails. An
    unsatisfiable plan must never reach the scheduler at all; the graph's own
    `validate_graph` is the second line of defence, not the first.
    """

    tasks: tuple[TaskSpec, ...] = ()

    @model_validator(mode="after")
    def _dependencies_are_satisfiable(self) -> TaskGraph:
        ids = [t.id for t in self.tasks]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate task ids: {duplicates}")
        known = set(ids)
        for task in self.tasks:
            unknown = sorted(set(task.depends_on) - known)
            if unknown:
                raise ValueError(f"task {task.id!r} depends on unknown task(s): {unknown}")
            if task.id in task.depends_on:
                raise ValueError(f"task {task.id!r} depends on itself")
        if cycle := _first_task_cycle(self.tasks):
            raise ValueError(f"task dependencies form a cycle: {' -> '.join(cycle)}")
        return self

    def ordered_ids(self) -> tuple[str, ...]:
        """Task ids in a dependency-respecting order. Useful for reporting and
        for tests; the scheduler never consults it - it derives readiness from
        graph edges, which is where dependency ordering actually executes."""
        remaining = {t.id: set(t.depends_on) for t in self.tasks}
        ordered: list[str] = []
        while remaining:
            ready = sorted(tid for tid, deps in remaining.items() if not deps - set(ordered))
            if not ready:  # pragma: no cover - the validator rejects cycles first
                break
            ordered.extend(ready)
            for tid in ready:
                del remaining[tid]
        return tuple(ordered)


def _first_task_cycle(tasks: tuple[TaskSpec, ...]) -> list[str] | None:
    """A dependency cycle as a readable path, or None. Iterative DFS with a
    colouring, so the error names the actual cycle rather than just asserting
    one exists - a decomposer being told "a -> b -> a" can fix it; one told
    "your graph has a cycle" often cannot."""
    adjacency = {t.id: list(t.depends_on) for t in tasks}
    white, grey, black = 0, 1, 2
    colour = dict.fromkeys(adjacency, white)
    for root in adjacency:
        if colour[root] != white:
            continue
        stack: list[tuple[str, int]] = [(root, 0)]
        path: list[str] = []
        while stack:
            node, index = stack[-1]
            if index == 0:
                colour[node] = grey
                path.append(node)
            if index < len(adjacency[node]):
                stack[-1] = (node, index + 1)
                child = adjacency[node][index]
                if colour.get(child) == grey:
                    return [*path[path.index(child) :], child]
                if colour.get(child) == white:
                    stack.append((child, 0))
            else:
                colour[node] = black
                path.pop()
                stack.pop()
    return None


# --- implementation -------------------------------------------------------


class FileChange(ArtifactModel):
    path: str
    change_kind: str = "modify"  # add | modify | delete
    content: str | None = None
    diff: str | None = None


class CodePatch(ArtifactModel):
    """Output of an implementer or repair agent. Applied only inside the
    sandbox (docs/05 section 3.5) - never to the real repository."""

    summary: str
    files: tuple[FileChange, ...] = ()


class DocsPatch(ArtifactModel):
    """Output of the docs agent: documentation changes, structured the same
    way as a `CodePatch` since both land through the same sandbox promotion
    path."""

    summary: str
    files: tuple[FileChange, ...] = ()


# --- testing and review ----------------------------------------------------


class TestSuite(ArtifactModel):
    summary: str
    files: tuple[FileChange, ...] = ()
    coverage_delta: float | None = None


class Finding(ArtifactModel):
    """One reviewer observation. `citation` is mandatory in spirit (docs/05
    section 3.3 requires the critic to cite file:line) though not enforced by
    this schema alone - L4 semantic validation checks that separately."""

    summary: str
    citation: str | None = None
    severity: str = "info"  # info | warning | high


class ReviewReport(ArtifactModel):
    findings: tuple[Finding, ...] = ()
    verdict: str = "pass"  # pass | concerns | block


class PolicyViolation(ArtifactModel):
    """A policy engine (Phase 3) finding, or - for now - a static security
    scan result. Distinct from the kernel's `POLICY_VIOLATION` event: this is
    the *content* an agent/tool produced; the event is the kernel's record
    that it happened."""

    rule: str
    severity: str = "medium"  # low | medium | high
    message: str
    node_id: str | None = None


# --- migration (docs/04 section 4: generate and classify, never apply) ----


class MigrationPlan(ArtifactModel):
    """Output of the migration agent. `classification` is never LLM-decided -
    it is filled in from `kernel.tools.migrations.classify_migration`'s
    deterministic verdict (safe | risky | destructive | opaque) on the
    materialized SQL, consistent with CLAUDE.md section 10's "deterministic
    validation is authoritative". The agent itself never applies this plan -
    `workflows/greenfield.yaml` gates `ef.database_update` behind a human
    approval node, the same mechanism gate1/gate2/gate3 already use."""

    migration_name: str
    summary: str
    sql: str
    classification: str  # safe | risky | destructive | opaque


# --- release --------------------------------------------------------------


class ReleaseReport(ArtifactModel):
    ready: bool
    checklist: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()


class RunSummary(ArtifactModel):
    """The final engineering summary (the brief's Feature 8), generated from
    the event log rather than hand-written - docs/05 section 6."""

    plan_and_rationale: str
    artifacts: tuple[str, ...] = ()  # artifact hashes referenced by the run
    risks: tuple[str, ...] = ()
    validation: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


# --- registry ---------------------------------------------------------

#: kind-string (as recorded in ARTIFACT_PRODUCED events and NodeSpec.produces)
#: -> the model it must validate against. Kept in sync with every workflow
#: YAML's `produces:` field by `tests/unit/test_contracts.py`.
CONTRACTS: dict[str, type[ArtifactModel]] = {
    "RequirementSpec": RequirementSpec,
    "AmbiguityRegister": AmbiguityRegister,
    "ImpactReport": ImpactReport,
    "ArchitectureDecision": ArchitectureDecision,
    "DesignSpec": DesignSpec,
    "SolutionSkeleton": SolutionSkeleton,
    "TaskGraph": TaskGraph,
    "CodePatch": CodePatch,
    "DocsPatch": DocsPatch,
    "TestSuite": TestSuite,
    "ReviewReport": ReviewReport,
    "PolicyViolation": PolicyViolation,
    "MigrationPlan": MigrationPlan,
    "ReleaseReport": ReleaseReport,
    "RunSummary": RunSummary,
}
