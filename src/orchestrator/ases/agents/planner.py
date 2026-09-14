"""Turns a decomposer's `TaskGraph` into a live implementation subgraph.

This is the piece that was missing. `WorkflowGraph.with_subgraph` has existed
and been tested since Phase 1, `SUBGRAPH_PROPOSED`/`ADMITTED`/`REJECTED` have
existed as event types, and `kernel/scheduler.py` said in its own docstring
that no executor outcome could yet cause a running scheduler to admit
anything. `workflows/greenfield.yaml` filled the gap with three hard-coded
nodes - `impl_domain`, `impl_api`, `impl_infrastructure` - and a comment
admitting they were stand-ins.

Live run `009ea59f-...` is what that cost. The decomposer proposed twenty
tasks across four layers, including five `app-*` tasks for
`UrlShortener.Application`. The static graph had nodes for three layers, so
the Application tasks were never dispatched, the project kept its scaffolded
`Class1.cs`, and every type Infrastructure and the API compiled against
(`IShortLinkRepository`, `IClock`, `IShortCodeGenerator`, `IShortLinkCache`,
`InsertShortLinkResult`) did not exist. The plan was correct; nothing
executed it. A fourth hard-coded node would have fixed that one run and left
the next architecture - `Web`/`Messaging`/`Worker`/`Persistence`, say - to
fail the same way.

**Shape admitted per task.** For task `T`, three nodes and four edges, plus
one more per declared dependency::

    impl:T ------> build:T --[on_failure]--> repair:T
                      |   <--[on_success]-------'
                      '---> impl_end

    and for each dependency D:   build:D ------> impl:T

**Why the dependency edge comes from `build:D`, not `impl:D`.** Waiting on
the dependency's *implementation* only proves its files were written;
waiting on its *build* proves they compile. Infrastructure implementing
`IShortLinkRepository` needs the Application project to contain that
interface and to have compiled, or `build:infrastructure` fails on a type
that was written but is not valid - a strictly worse failure to hand a
repair agent, since the fault is then in a project it has no business
editing.

**Where parallelism comes from.** Nowhere in this module: it emits edges,
and `Scheduler._compute_ready` dispatches every ready node in one
`asyncio.gather`. Two tasks with no path between them become ready in the
same pass and genuinely run together; a task with unsatisfied dependency
edges is not ready, so it waits. Ordering is a property of the edges the
decomposer's `depends_on` produced - never of list order, task naming, or
which node happened to finish first.

**What the static YAML keeps.** `impl_start` and `impl_end`, two barriers.
`impl_start -> impl_end` remains as a direct edge so the file is a valid
graph on its own (`validate_graph` runs at load, before any task exists) and
so an empty `TaskGraph` degrades to "no implementation work" rather than to
a dangling graph. `impl_end` joins `all`, so it is satisfied only when
`impl_start` *and* every admitted `build:T` have succeeded - which is also
what keeps a failed build from letting the run past it.
"""

from __future__ import annotations

from typing import Final

from ases.context.lineage import UnknownArtifactError
from ases.context.retriever import ContextRetriever, NoArtifactFromNodeError
from ases.contracts.artifacts import CodePatch, SolutionSkeleton, TaskGraph, TaskSpec
from ases.kernel.graph import Edge, EdgeCondition, JoinPolicy, NodeKind, NodeSpec
from ases.kernel.scheduler import SubgraphProposal
from ases.kernel.state import RunState

#: Node-id prefixes for the three positions each task materializes into.
#: A structural convention the *orchestrator* imposes on ids it generates
#: itself - deliberately not an inference over model-authored names, which is
#: the thing this module exists to stop doing. `agents/implementer.py` and
#: `agents/wiring.py` parse ids back through `task_id_of`, never by matching
#: substrings of a project name.
IMPL_PREFIX: Final = "impl:"
BUILD_PREFIX: Final = "build:"
REPAIR_PREFIX: Final = "repair:"

#: The only `TaskSpec.kind` this provider routes. A task of any other kind is
#: rejected at admission rather than silently skipped: silently skipping work
#: a decomposer asked for is precisely the failure mode of the static graph
#: this replaces.
IMPLEMENTATION_KIND: Final = "implementation"

#: How many build attempts one task gets before its cycle budget is spent.
#: Matches what the static `build_*` nodes declared.
BUILD_CYCLE_BUDGET: Final = 2

_IMPL_TIMEOUT_SECONDS: Final = 300.0
_BUILD_TIMEOUT_SECONDS: Final = 300.0


class SubgraphPlanError(ValueError):
    """The `TaskGraph` cannot be turned into an executable subgraph.

    Distinct from `GraphError` (the *graph* refused the proposal) because it
    says something different: the plan is internally consistent but does not
    fit the solution it is planning for - naming a project the scaffold never
    created, or leaving one nobody implements.
    """


def impl_node_id(task_id: str) -> str:
    return f"{IMPL_PREFIX}{task_id}"


def build_node_id(task_id: str) -> str:
    return f"{BUILD_PREFIX}{task_id}"


def repair_node_id(task_id: str) -> str:
    return f"{REPAIR_PREFIX}{task_id}"


def task_id_of(node_id: str) -> str | None:
    """The task a generated node belongs to, or `None` for a static node.

    The inverse of the three id builders above, and the only supported way to
    get from a node id back to a task - see the prefix constants' note.
    """
    for prefix in (IMPL_PREFIX, BUILD_PREFIX, REPAIR_PREFIX):
        if node_id.startswith(prefix):
            return node_id[len(prefix) :]
    return None


def is_repair_node(node_id: str) -> bool:
    return node_id.startswith(REPAIR_PREFIX)


def task_graph_of(state: RunState, source_node_id: str = "decompose") -> TaskGraph | None:
    """The run's `TaskGraph`, or `None` if the decomposer has not run."""
    try:
        artifact = ContextRetriever(state).fetch_latest_from(source_node_id)
    except (NoArtifactFromNodeError, UnknownArtifactError):
        return None
    return artifact if isinstance(artifact, TaskGraph) else None


def task_of(state: RunState, node_id: str, *, source_node_id: str = "decompose") -> TaskSpec | None:
    """The `TaskSpec` a generated node implements, read back from the run's
    own `TaskGraph` artifact. `None` for a static node, or before the
    decomposer has produced anything."""
    task_id = task_id_of(node_id)
    if task_id is None:
        return None
    graph = task_graph_of(state, source_node_id)
    if graph is None:
        return None
    return next((t for t in graph.tasks if t.id == task_id), None)


def implementation_node_ids(state: RunState) -> tuple[str, ...]:
    """Every node in this run that produced a `CodePatch`, in id order.

    The answer to "what did the implementation stage write?", derived from
    the run rather than declared. Each of `agents/docs.py`, `reviewer.py` and
    `tester.py` used to carry its own hard-coded
    `IMPLEMENTATION_NODE_IDS = ("impl_domain", "impl_api")` - three copies of
    the same idea, all three of which omitted `impl_infrastructure` even
    after the static graph grew it. The reviewer, the test generator and the
    documentation agent have therefore never seen a line of infrastructure
    code in any run this system has performed. Deriving the list fixes that
    and makes runtime-admitted tasks visible on the same terms.

    Includes `repair:<task>` nodes: a repair's patch is the current state of
    that task's code, and reviewing the pre-repair version would be
    reviewing code that no longer exists in the sandbox.
    """
    retriever = ContextRetriever(state)
    found: list[str] = []
    for node_id in sorted(state.nodes):
        try:
            artifact = retriever.fetch_latest_from(node_id)
        except (NoArtifactFromNodeError, UnknownArtifactError):
            continue
        if isinstance(artifact, CodePatch):
            found.append(node_id)
    return tuple(found)


def implementation_patches(state: RunState) -> tuple[CodePatch, ...]:
    """The latest `CodePatch` from each node `implementation_node_ids` names."""
    retriever = ContextRetriever(state)
    patches: list[CodePatch] = []
    for node_id in implementation_node_ids(state):
        artifact = retriever.fetch_latest_from(node_id)
        if isinstance(artifact, CodePatch):
            patches.append(artifact)
    return tuple(patches)


def check_plan_fits_solution(task_graph: TaskGraph, skeleton: SolutionSkeleton | None) -> None:
    """Structural validation of a plan against the solution it plans for.

    Raises `SubgraphPlanError` describing the first problem found, in terms a
    decomposer can act on - `agents/decompose.py` feeds exactly this message
    back for one bounded retry, and `TaskGraphSubgraphProvider.propose` calls
    it again as the authoritative gate before anything is admitted. Stating
    it once and using it in both places is what keeps "what the agent was
    told to fix" and "what the runtime will actually accept" from drifting
    apart.

    Two questions the graph itself cannot answer, because neither is about
    graph shape: does every task name a project that actually exists, and
    does every project the architecture called for have someone implementing
    it? The second is the check that would have caught live run
    `009ea59f-...` before it spent a single implementation call - rather than
    three builds, two repairs and a safe-stop later, with the halt reason
    naming the wrong node.
    """
    for task in task_graph.tasks:
        if task.kind != IMPLEMENTATION_KIND:
            raise SubgraphPlanError(
                f"task {task.id!r} has kind {task.kind!r}; only "
                f"{IMPLEMENTATION_KIND!r} tasks can be executed, and silently "
                "skipping a task the decomposer asked for is not an option"
            )
        if not task.component:
            raise SubgraphPlanError(
                f"task {task.id!r} names no component; every implementation task "
                "must say which project it writes into"
            )

    if skeleton is None or not skeleton.projects:
        return
    projects = set(skeleton.projects)
    for task in task_graph.tasks:
        if task.component not in projects:
            raise SubgraphPlanError(
                f"task {task.id!r} targets component {task.component!r}, which is not "
                f"one of the scaffolded projects: {sorted(projects)}"
            )
    covered = {t.component for t in task_graph.tasks}
    uncovered = sorted(projects - covered - _exempt_projects(skeleton))
    if uncovered:
        raise SubgraphPlanError(
            f"no implementation task targets {uncovered} - the scaffold created "
            f"{len(projects)} project(s) and the plan implements {len(covered)}. A "
            "project the architecture called for but nobody implements compiles as an "
            "empty placeholder and fails every project that depends on it"
        )


def _exempt_projects(skeleton: SolutionSkeleton) -> set[str]:
    """Projects allowed to carry no implementation task.

    One category only: a generated test project that no frozen interface
    mentions has nothing another task can depend on, so leaving it empty
    cannot break a build. Deliberately narrow - the point of the coverage
    check is that an unimplemented project is normally a bug, and the one
    this system actually shipped cost a whole run.
    """
    mentioned = " ".join(
        f"{i.signature} {i.namespace} {i.project}" for i in skeleton.frozen_interfaces
    )
    return {
        project
        for project in skeleton.projects
        if project.rsplit(".", 1)[-1].lower().endswith("tests") and project not in mentioned
    }


def _same_component_predecessors(task_graph: TaskGraph) -> dict[str, str]:
    """For each task, the task that writes the same project immediately
    before it, if any.

    **Why a project needs a write lock.** Live run `91229361-...` planned
    three tasks - `infra-persistence`, `infra-cache`, `infra-codegen-clock` -
    all with `component: UrlShortener.Infrastructure` and all with the same
    single dependency, so all three were dispatched together. Three
    consequences, all bad, none of which the task graph itself is wrong
    about:

    - Three `build:<task>` nodes ran the identical `dotnet build
      UrlShortener.Infrastructure`. A per-task build gate is supposed to say
      "this task compiles"; run concurrently against one project it can only
      say "the union of three tasks compiles", which is a different claim
      and is the one that failed.
    - One task's file changed how another task's file compiled.
      `infra-persistence` emitted a `_NamespaceCompatibility.cs` declaring
      empty `namespace UrlShortener.Application.Ports { }` and friends, to
      keep its own speculative `using` directives resolving. That made
      `infra-cache`'s *wrong* `using UrlShortener.Application.Ports;` compile
      instead of failing at the import, so the error surfaced fifteen lines
      later as "type not found" against the class declaration.
    - All three builds failed on one error and all three opened a repair
      position at once: three agents rewriting one project, each reading a
      diagnostic caused by a different task's file.

    Ordering is taken from `TaskGraph.ordered_ids()`, a topological order, not
    from the order the tasks happen to be listed in. That matters for more
    than tidiness: every edge this adds runs forward along a topological
    order, and every dependency edge already does, so the admitted subgraph
    stays acyclic by construction. Chaining in listed order does not - a plan
    listing two tasks of one component with a task of another component
    depending between them would close a loop, and `with_subgraph` would
    reject the whole admission.

    This is the only serialization the planner imposes. Tasks in different
    projects with no declared dependency between them are still dispatched
    together; what cannot overlap is two tasks writing the same compilation
    unit, which cannot be validated independently anyway.
    """
    previous_by_component: dict[str, str] = {}
    predecessors: dict[str, str] = {}
    by_id = {task.id: task for task in task_graph.tasks}
    for task_id in task_graph.ordered_ids():
        component = by_id[task_id].component
        if (previous := previous_by_component.get(component)) is not None:
            predecessors[task_id] = previous
        previous_by_component[component] = task_id
    return predecessors


class TaskGraphSubgraphProvider:
    """`kernel.scheduler.SubgraphProvider` over the decomposer's output.

    Constructed by `agents.wiring` with the node ids the workflow actually
    uses, so nothing here is tied to `greenfield.yaml`'s particular naming: a
    second workflow with a differently-named decompose stage supplies its own.
    """

    def __init__(
        self,
        *,
        source_node_id: str = "decompose",
        skeleton_node_id: str = "scaffold",
        start_node_id: str = "impl_start",
        end_node_id: str = "impl_end",
        implementer_handler: str = "implementer",
        build_handler: str = "dotnet_build",
    ) -> None:
        self.source_node_id = source_node_id
        self.skeleton_node_id = skeleton_node_id
        self.start_node_id = start_node_id
        self.end_node_id = end_node_id
        self.implementer_handler = implementer_handler
        self.build_handler = build_handler

    def propose(self, node: NodeSpec, state: RunState) -> SubgraphProposal | None:
        if node.id != self.source_node_id:
            return None
        task_graph = task_graph_of(state, self.source_node_id)
        if task_graph is None or not task_graph.tasks:
            # An empty plan is a legitimate (if unusual) answer, and the
            # static `impl_start -> impl_end` edge already expresses it. It
            # is not an error to admit nothing.
            return None
        self._check_plan_fits_the_solution(task_graph, state)
        return SubgraphProposal(
            nodes=self._nodes_for(task_graph), edges=self._edges_for(task_graph)
        )

    def _check_plan_fits_the_solution(self, task_graph: TaskGraph, state: RunState) -> None:
        check_plan_fits_solution(task_graph, self._skeleton(state))

    def _skeleton(self, state: RunState) -> SolutionSkeleton | None:
        try:
            artifact = ContextRetriever(state).fetch_latest_from(self.skeleton_node_id)
        except (NoArtifactFromNodeError, UnknownArtifactError):
            return None
        return artifact if isinstance(artifact, SolutionSkeleton) else None

    def _nodes_for(self, task_graph: TaskGraph) -> tuple[NodeSpec, ...]:
        nodes: list[NodeSpec] = []
        for task in task_graph.tasks:
            nodes.append(
                NodeSpec(
                    id=impl_node_id(task.id),
                    kind=NodeKind.AGENT,
                    handler=self.implementer_handler,
                    produces="CodePatch",
                    # `join` stays at its `all` default: every dependency's
                    # build must have succeeded, not merely one of them.
                    timeout_seconds=_IMPL_TIMEOUT_SECONDS,
                    description=f"{task.component}: {task.description}",
                )
            )
            nodes.append(
                NodeSpec(
                    id=build_node_id(task.id),
                    kind=NodeKind.TOOL,
                    handler=self.build_handler,
                    # `any`: ready on the first arrival from `impl:T`, and
                    # again after `repair:T` succeeds. Bounded by the budget.
                    join=JoinPolicy.ANY,
                    cycle_budget=BUILD_CYCLE_BUDGET,
                    timeout_seconds=_BUILD_TIMEOUT_SECONDS,
                    description=f"dotnet build {task.component}",
                )
            )
            nodes.append(
                NodeSpec(
                    id=repair_node_id(task.id),
                    kind=NodeKind.AGENT,
                    handler=self.implementer_handler,
                    produces="CodePatch",
                    timeout_seconds=_IMPL_TIMEOUT_SECONDS,
                    description=f"repair {task.component} after a build failure",
                )
            )
        return tuple(nodes)

    def _edges_for(self, task_graph: TaskGraph) -> tuple[Edge, ...]:
        edges: list[Edge] = []
        predecessor_in_component = _same_component_predecessors(task_graph)
        for task in task_graph.tasks:
            impl = impl_node_id(task.id)
            build = build_node_id(task.id)
            repair = repair_node_id(task.id)
            # Two sources of incoming edges, and the node joins `all` over
            # both: the task's own declared dependencies, and the task that
            # holds this component's write lock before it.
            incoming = {build_node_id(d) for d in task.depends_on}
            if (previous := predecessor_in_component.get(task.id)) is not None:
                incoming.add(build_node_id(previous))
            if incoming:
                # Wait on each source's *build*, not its implementation -
                # see the module docstring.
                edges.extend(Edge(source=source, target=impl) for source in sorted(incoming))
            else:
                edges.append(Edge(source=self.start_node_id, target=impl))
            edges.append(Edge(source=impl, target=build))
            edges.append(Edge(source=build, target=repair, condition=EdgeCondition.ON_FAILURE))
            # `on_success`, never `always`: a repair that produced no patch
            # must not re-trigger a build of unchanged source and spend that
            # build's cycle budget - see `workflows/greenfield.yaml`'s note.
            edges.append(Edge(source=repair, target=build, condition=EdgeCondition.ON_SUCCESS))
            edges.append(Edge(source=build, target=self.end_node_id))
        return tuple(edges)
