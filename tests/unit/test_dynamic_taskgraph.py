"""The decomposer's TaskGraph, executed as a live subgraph.

Covers the path Architecture -> Decomposer -> TaskGraph -> runtime subgraph ->
Scheduler, which replaced `workflows/greenfield.yaml`'s hard-coded
`impl_domain`/`impl_api`/`impl_infrastructure` fan-out. The failure being
regressed against is live run `009ea59f-...`: the decomposer proposed five
`app-*` tasks for `UrlShortener.Application`, the static graph had no node
for that layer, the project kept its scaffolded `Class1.cs`, and every type
Infrastructure and the API compiled against failed to resolve.

The layer names in these tests are fixtures, not fixed points - the whole
point is that nothing in the orchestrator knows what "Application" is.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from ases.agents.planner import (
    SubgraphPlanError,
    TaskGraphSubgraphProvider,
    build_node_id,
    impl_node_id,
    implementation_node_ids,
    repair_node_id,
    task_id_of,
    task_of,
)
from ases.contracts.artifacts import (
    CodePatch,
    FileChange,
    FrozenInterface,
    SolutionSkeleton,
    TaskGraph,
    TaskSpec,
)
from ases.kernel.events import EventType
from ases.kernel.failures import FailureKind
from ases.kernel.gates import BudgetEntryGate, BudgetLimits
from ases.kernel.graph import (
    Edge,
    EdgeCondition,
    JoinPolicy,
    NodeKind,
    NodeSpec,
    WorkflowGraph,
)
from ases.kernel.scheduler import NodeExecutionOutcome, NodeExecutor, Scheduler
from ases.kernel.state import ArtifactRecord, NodeState, NodeStatus, RunState, RunStatus
from ases.kernel.store.jsonl import JsonlEventStore

GENEROUS = BudgetLimits(max_tokens=10_000_000, max_usd=1_000.0, max_wallclock_seconds=3600)

#: A four-layer plan whose dependency shape is the one the URL shortener
#: actually needs, and which the static graph could not express: Application
#: after Domain, then Infrastructure and API *both* after Application and
#: independent of each other.
FOUR_LAYER_TASKS = TaskGraph(
    tasks=(
        TaskSpec(id="domain", description="entities", component="Shop.Domain"),
        TaskSpec(
            id="application",
            description="ports and use cases",
            component="Shop.Application",
            depends_on=("domain",),
        ),
        TaskSpec(
            id="infrastructure",
            description="EF Core adapters",
            component="Shop.Infrastructure",
            depends_on=("application",),
        ),
        TaskSpec(
            id="api",
            description="controllers",
            component="Shop.Api",
            depends_on=("application",),
        ),
    )
)

FOUR_LAYER_SKELETON = SolutionSkeleton(
    projects=("Shop.Domain", "Shop.Application", "Shop.Infrastructure", "Shop.Api"),
    frozen_interfaces=(
        FrozenInterface(
            signature="public interface IShortLinkRepository { }",
            namespace="Shop.Application",
            project="Shop.Application",
        ),
    ),
)


def _state_with(tasks: TaskGraph, skeleton: SolutionSkeleton | None) -> RunState:
    """A folded state as it stands the moment `decompose` has just succeeded."""
    state = RunState(run_id=uuid4())
    state.artifacts["h-tasks"] = ArtifactRecord(
        artifact_hash="h-tasks",
        kind="TaskGraph",
        node_id="decompose",
        produced_at=datetime.now(UTC),
    )
    state.artifact_content["h-tasks"] = tasks.model_dump(mode="json")
    state.nodes["decompose"] = NodeState(node_id="decompose", produced=("h-tasks",))
    if skeleton is not None:
        state.artifacts["h-skel"] = ArtifactRecord(
            artifact_hash="h-skel",
            kind="SolutionSkeleton",
            node_id="scaffold",
            produced_at=datetime.now(UTC),
        )
        state.artifact_content["h-skel"] = skeleton.model_dump(mode="json")
        state.nodes["scaffold"] = NodeState(node_id="scaffold", produced=("h-skel",))
    return state


def _decompose_node() -> NodeSpec:
    return NodeSpec(id="decompose", kind=NodeKind.AGENT, handler="decompose", produces="TaskGraph")


# --- TaskGraph dependencies are explicit and enforced ----------------------


def test_dependencies_must_name_tasks_that_exist() -> None:
    with pytest.raises(ValueError, match="depends on unknown task"):
        TaskGraph(tasks=(TaskSpec(id="a", description="x", depends_on=("ghost",)),))


def test_a_dependency_cycle_is_rejected_and_named() -> None:
    """Named, not merely detected: a decomposer told "a -> b -> a" can fix
    its plan; one told "your graph has a cycle" often cannot."""
    with pytest.raises(ValueError, match=r"cycle: a -> b -> a"):
        TaskGraph(
            tasks=(
                TaskSpec(id="a", description="x", depends_on=("b",)),
                TaskSpec(id="b", description="y", depends_on=("a",)),
            )
        )


def test_duplicate_task_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate task ids"):
        TaskGraph(tasks=(TaskSpec(id="a", description="x"), TaskSpec(id="a", description="y")))


def test_dependency_order_comes_from_depends_on_not_list_order() -> None:
    """The plan is declared back-to-front on purpose."""
    reversed_plan = TaskGraph(tasks=tuple(reversed(FOUR_LAYER_TASKS.tasks)))
    order = reversed_plan.ordered_ids()
    assert order.index("domain") < order.index("application")
    assert order.index("application") < order.index("infrastructure")
    assert order.index("application") < order.index("api")


# --- structural validation, before anything expensive runs -----------------


def test_a_project_with_no_implementation_task_is_rejected_at_admission() -> None:
    """The check that would have caught live run `009ea59f-...` before a
    single implementation call, instead of three builds and two repairs
    later with the halt reason naming the wrong node."""
    # Exactly live run `009ea59f-...`'s plan: Infrastructure and the API are
    # planned, the layer they both compile against is not. Built in one go
    # because `TaskGraph` will not construct with a dangling `depends_on` -
    # so the omission has to be a *silent* one, which is what makes it the
    # dangerous case.
    plan_missing_application = TaskGraph(
        tasks=(
            TaskSpec(id="domain", description="entities", component="Shop.Domain"),
            TaskSpec(
                id="infrastructure",
                description="EF Core adapters",
                component="Shop.Infrastructure",
                depends_on=("domain",),
            ),
            TaskSpec(
                id="api",
                description="controllers",
                component="Shop.Api",
                depends_on=("domain",),
            ),
        )
    )
    state = _state_with(plan_missing_application, FOUR_LAYER_SKELETON)

    with pytest.raises(SubgraphPlanError, match=r"Shop\.Application"):
        TaskGraphSubgraphProvider().propose(_decompose_node(), state)


def test_a_task_targeting_a_project_that_was_never_scaffolded_is_rejected() -> None:
    plan = TaskGraph(tasks=(TaskSpec(id="ghost", description="x", component="Shop.DoesNotExist"),))
    state = _state_with(plan, FOUR_LAYER_SKELETON)

    with pytest.raises(SubgraphPlanError, match="not one of the scaffolded projects"):
        TaskGraphSubgraphProvider().propose(_decompose_node(), state)


def test_a_task_that_names_no_component_is_rejected() -> None:
    state = _state_with(TaskGraph(tasks=(TaskSpec(id="a", description="x"),)), FOUR_LAYER_SKELETON)

    with pytest.raises(SubgraphPlanError, match="names no component"):
        TaskGraphSubgraphProvider().propose(_decompose_node(), state)


def test_a_task_of_an_unroutable_kind_is_rejected_rather_than_skipped() -> None:
    """Silently skipping work the decomposer asked for is the exact failure
    mode this whole change exists to remove."""
    plan = TaskGraph(
        tasks=(TaskSpec(id="a", description="x", component="Shop.Domain", kind="benchmark"),)
    )
    state = _state_with(plan, FOUR_LAYER_SKELETON)

    with pytest.raises(SubgraphPlanError, match="only 'implementation' tasks"):
        TaskGraphSubgraphProvider().propose(_decompose_node(), state)


def test_a_generated_test_project_may_go_unimplemented() -> None:
    skeleton = SolutionSkeleton(
        projects=("Shop.Domain", "Shop.ArchitectureTests"), frozen_interfaces=()
    )
    plan = TaskGraph(tasks=(TaskSpec(id="d", description="x", component="Shop.Domain"),))

    proposal = TaskGraphSubgraphProvider().propose(_decompose_node(), _state_with(plan, skeleton))

    assert proposal is not None


# --- the proposal's shape --------------------------------------------------


def test_each_task_becomes_an_implement_build_repair_triple() -> None:
    proposal = TaskGraphSubgraphProvider().propose(
        _decompose_node(), _state_with(FOUR_LAYER_TASKS, FOUR_LAYER_SKELETON)
    )
    assert proposal is not None
    ids = {n.id for n in proposal.nodes}
    for task_id in ("domain", "application", "infrastructure", "api"):
        assert {impl_node_id(task_id), build_node_id(task_id), repair_node_id(task_id)} <= ids
    assert task_id_of("build:application") == "application"


def test_a_dependency_becomes_an_edge_from_the_dependency_s_build() -> None:
    """From `build:D`, not `impl:D`: waiting on the implementation only
    proves the files were written, waiting on the build proves they compile."""
    proposal = TaskGraphSubgraphProvider().propose(
        _decompose_node(), _state_with(FOUR_LAYER_TASKS, FOUR_LAYER_SKELETON)
    )
    assert proposal is not None
    incoming = {
        (e.source, e.target) for e in proposal.edges if e.target == impl_node_id("infrastructure")
    }
    assert incoming == {(build_node_id("application"), impl_node_id("infrastructure"))}


def test_the_proposal_is_accepted_by_the_real_greenfield_graph(
    greenfield_graph: WorkflowGraph,
) -> None:
    """`with_subgraph` re-validates the whole candidate: no dangling edge, no
    unreachable node, no node that cannot reach a terminal, no unbounded
    cycle."""
    proposal = TaskGraphSubgraphProvider().propose(
        _decompose_node(), _state_with(FOUR_LAYER_TASKS, FOUR_LAYER_SKELETON)
    )
    assert proposal is not None

    admitted = greenfield_graph.with_subgraph(proposal.nodes, proposal.edges)

    assert impl_node_id("application") in admitted.by_id
    # Every build feeds the barrier that gates the rest of the lifecycle.
    for task_id in ("domain", "application", "infrastructure", "api"):
        assert any(
            e.source == build_node_id(task_id) and e.target == "impl_end" for e in admitted.edges
        )


@pytest.fixture
def greenfield_graph() -> WorkflowGraph:
    import ases

    return WorkflowGraph.from_yaml(Path(ases.__file__).parent / "workflows" / "greenfield.yaml")


# --- execution: ordering, parallelism, and the failed-build barrier --------


class _RecordingExecutor:
    """Records when each node ran and how many ran at once."""

    def __init__(self, fail_nodes: frozenset[str] = frozenset()) -> None:
        self.started: list[str] = []
        self.max_concurrent = 0
        self._live = 0
        self._fail = fail_nodes

    async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome:
        self.started.append(node.id)
        self._live += 1
        self.max_concurrent = max(self.max_concurrent, self._live)
        try:
            # Two hops through the event loop, so genuinely concurrent calls
            # overlap here and sequential ones cannot.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        finally:
            self._live -= 1
        if node.id in self._fail:
            return NodeExecutionOutcome(
                ok=False,
                failure_kind=FailureKind.BUILD_FAILURE,
                error="error CS0246: type or namespace not found",
            )
        if node.kind is NodeKind.AGENT:
            return NodeExecutionOutcome(
                ok=True,
                artifact_kind="CodePatch",
                artifact_payload=CodePatch(
                    summary=f"wrote {node.id}",
                    files=(FileChange(path=f"{node.id}.cs", content="// x"),),
                ).model_dump(mode="json"),
            )
        return NodeExecutionOutcome(ok=True)


class _DecomposeExecutor:
    def __init__(self, tasks: TaskGraph) -> None:
        self._tasks = tasks

    async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome:
        return NodeExecutionOutcome(
            ok=True, artifact_kind="TaskGraph", artifact_payload=self._tasks.model_dump(mode="json")
        )


def _execution_stage_graph() -> WorkflowGraph:
    """The generic execution stage in isolation: exactly the shape
    `greenfield.yaml` declares between `decompose` and the rest."""
    return WorkflowGraph(
        name="execution_stage",
        entry=("scaffold",),
        nodes=(
            NodeSpec(
                id="scaffold", kind=NodeKind.AGENT, handler="scaffold", produces="SolutionSkeleton"
            ),
            NodeSpec(
                id="decompose", kind=NodeKind.AGENT, handler="decompose", produces="TaskGraph"
            ),
            NodeSpec(id="impl_start", kind=NodeKind.BARRIER, join=JoinPolicy.ALL),
            NodeSpec(id="impl_end", kind=NodeKind.BARRIER, join=JoinPolicy.ALL),
            NodeSpec(id="done", kind=NodeKind.TERMINAL),
        ),
        edges=(
            Edge(source="scaffold", target="decompose"),
            Edge(source="decompose", target="impl_start"),
            Edge(source="impl_start", target="impl_end"),
            Edge(source="impl_end", target="done"),
        ),
    )


class _SkeletonExecutor:
    async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome:
        return NodeExecutionOutcome(
            ok=True,
            artifact_kind="SolutionSkeleton",
            artifact_payload=FOUR_LAYER_SKELETON.model_dump(mode="json"),
        )


async def _run_stage(
    tmp_path: Path,
    *,
    tasks: TaskGraph = FOUR_LAYER_TASKS,
    fail_nodes: frozenset[str] = frozenset(),
) -> tuple[RunState, _RecordingExecutor, JsonlEventStore]:
    worker = _RecordingExecutor(fail_nodes)
    executors: dict[str, NodeExecutor] = {
        "scaffold": _SkeletonExecutor(),
        "decompose": _DecomposeExecutor(tasks),
        "implementer": worker,
        "dotnet_build": worker,
    }
    store = JsonlEventStore(tmp_path / "events", fsync=False)
    state = await Scheduler(
        _execution_stage_graph(),
        store,
        executors,
        entry_gate=BudgetEntryGate(GENEROUS),
        subgraphs=TaskGraphSubgraphProvider(),
    ).run(uuid4())
    return state, worker, store


async def test_the_whole_plan_executes_and_the_run_completes(tmp_path: Path) -> None:
    state, _, _ = await _run_stage(tmp_path)

    assert state.status is RunStatus.COMPLETED
    for task_id in ("domain", "application", "infrastructure", "api"):
        assert state.nodes[impl_node_id(task_id)].status is NodeStatus.SUCCEEDED
        assert state.nodes[build_node_id(task_id)].status is NodeStatus.SUCCEEDED


async def test_a_dependent_task_never_starts_before_its_dependency_has_built(
    tmp_path: Path,
) -> None:
    """Ordering comes from the edges `depends_on` produced, not from list
    order, task naming or which node happened to finish first."""
    _, worker, _ = await _run_stage(tmp_path)
    order = worker.started

    assert order.index(build_node_id("domain")) < order.index(impl_node_id("application"))
    assert order.index(build_node_id("application")) < order.index(impl_node_id("infrastructure"))
    assert order.index(build_node_id("application")) < order.index(impl_node_id("api"))


async def test_infrastructure_cannot_start_before_the_application_layer_exists(
    tmp_path: Path,
) -> None:
    """The live failure, stated as an invariant. `Shop.Infrastructure`
    implements interfaces `Shop.Application` declares; starting it first is
    how `IShortLinkRepository`, `IClock`, `IShortCodeGenerator` and
    `IShortLinkCache` came to be referenced by code and defined by nobody."""
    state, _, store = await _run_stage(tmp_path)
    events = [e async for e in store.read(state.run_id)]
    seq_of = {
        (e.type, e.node_id): e.seq
        for e in events
        if e.type in (EventType.NODE_STARTED, EventType.NODE_SUCCEEDED)
    }

    application_built = seq_of[(EventType.NODE_SUCCEEDED, build_node_id("application"))]
    infrastructure_started = seq_of[(EventType.NODE_STARTED, impl_node_id("infrastructure"))]
    assert application_built < infrastructure_started


async def test_independent_tasks_run_concurrently(tmp_path: Path) -> None:
    """Dependency safety must not be bought by serialising everything.
    `infrastructure` and `api` both depend on `application` and on nothing
    else, so once it builds they are dispatched in the same pass."""
    _, worker, _ = await _run_stage(tmp_path)

    assert worker.max_concurrent >= 2


async def test_a_strictly_linear_plan_never_runs_two_things_at_once(tmp_path: Path) -> None:
    """The control for the test above: with a chain, the same machinery must
    produce no concurrency at all - proving the observed overlap came from
    the declared dependencies and not from the scheduler ignoring them."""
    linear = TaskGraph(
        tasks=(
            TaskSpec(id="domain", description="x", component="Shop.Domain"),
            TaskSpec(
                id="application",
                description="x",
                component="Shop.Application",
                depends_on=("domain",),
            ),
            TaskSpec(
                id="infrastructure",
                description="x",
                component="Shop.Infrastructure",
                depends_on=("application",),
            ),
            TaskSpec(
                id="api", description="x", component="Shop.Api", depends_on=("infrastructure",)
            ),
        )
    )
    _, worker, _ = await _run_stage(tmp_path, tasks=linear)

    assert worker.max_concurrent == 1


async def test_a_failed_task_build_stops_the_execution_stage(tmp_path: Path) -> None:
    """`impl_end` joins `all` over every admitted build, so one build that
    never goes green cannot let the run reach the rest of the lifecycle -
    the same property the static graph had, preserved by construction."""
    state, _, _ = await _run_stage(tmp_path, fail_nodes=frozenset({build_node_id("application")}))

    assert state.status is not RunStatus.COMPLETED
    assert state.status_of("impl_end") is NodeStatus.PENDING
    assert state.status_of("done") is NodeStatus.PENDING
    # ...and the tasks that depended on it never started.
    assert state.status_of(impl_node_id("infrastructure")) is NodeStatus.PENDING
    assert state.status_of(impl_node_id("api")) is NodeStatus.PENDING


async def test_a_failed_build_routes_to_that_task_s_own_repair(tmp_path: Path) -> None:
    _, worker, _ = await _run_stage(tmp_path, fail_nodes=frozenset({build_node_id("application")}))

    assert repair_node_id("application") in worker.started
    # Scoped: a sibling task's repair position was never dispatched.
    assert repair_node_id("domain") not in worker.started


# --- admission is recorded, and survives a resume --------------------------


async def test_admission_is_recorded_in_the_event_log(tmp_path: Path) -> None:
    state, _, store = await _run_stage(tmp_path)
    events = [e async for e in store.read(state.run_id)]

    admitted = [e for e in events if e.type is EventType.SUBGRAPH_ADMITTED]
    assert len(admitted) == 1
    assert admitted[0].node_id == "decompose"
    assert impl_node_id("application") in admitted[0].payload["node_ids"]
    # Recorded whole, not just by id - this is what makes a resume possible.
    assert admitted[0].payload["nodes"]
    assert admitted[0].payload["edges"]


async def test_a_resumed_run_rebuilds_the_subgraph_it_was_executing(tmp_path: Path) -> None:
    """Without this, resuming a run that got past `decompose` would rebuild
    the static YAML graph and then fold a state full of nodes that graph has
    never heard of."""
    state, _, store = await _run_stage(tmp_path)

    # A brand-new Scheduler over the *static* graph, as a fresh process gets it.
    resumed = Scheduler(
        _execution_stage_graph(),
        store,
        {},
        entry_gate=BudgetEntryGate(GENEROUS),
        subgraphs=TaskGraphSubgraphProvider(),
    )
    state_again = await resumed.run(state.run_id)

    assert state_again.status is RunStatus.COMPLETED
    assert impl_node_id("application") in resumed.graph.by_id
    assert build_node_id("api") in resumed.graph.by_id


async def test_a_rejected_proposal_fails_the_proposing_node_and_admits_nothing(
    tmp_path: Path,
) -> None:
    """Bounded autonomy: a proposal the graph refuses never executes, and the
    refusal is recorded rather than silently dropped."""

    class _ProposesADanglingEdge:
        def propose(self, node: NodeSpec, state: RunState) -> object:
            from ases.kernel.scheduler import SubgraphProposal

            if node.id != "decompose":
                return None
            return SubgraphProposal(
                nodes=(NodeSpec(id="orphan", kind=NodeKind.AGENT, handler="implementer"),),
                edges=(Edge(source="orphan", target="nonexistent"),),
            )

    executors: dict[str, NodeExecutor] = {
        "scaffold": _SkeletonExecutor(),
        "decompose": _DecomposeExecutor(FOUR_LAYER_TASKS),
        "implementer": _RecordingExecutor(),
        "dotnet_build": _RecordingExecutor(),
    }
    store = JsonlEventStore(tmp_path / "events", fsync=False)
    state = await Scheduler(
        _execution_stage_graph(),
        store,
        executors,
        entry_gate=BudgetEntryGate(GENEROUS),
        subgraphs=_ProposesADanglingEdge(),  # type: ignore[arg-type]
    ).run(uuid4())

    assert state.status is RunStatus.FAILED
    assert state.nodes["decompose"].status is NodeStatus.FAILED
    assert "orphan" not in state.nodes
    events = [e async for e in store.read(state.run_id)]
    assert any(e.type is EventType.SUBGRAPH_REJECTED for e in events)
    failed = [e for e in events if e.type is EventType.NODE_FAILED]
    assert failed[-1].payload["failure_kind"] == FailureKind.ORCHESTRATION_FAILURE.value


async def test_an_empty_plan_admits_nothing_and_still_completes(tmp_path: Path) -> None:
    """`impl_start -> impl_end` is what makes this degrade cleanly instead of
    hanging on a barrier nothing will ever satisfy."""
    state, worker, _ = await _run_stage(tmp_path, tasks=TaskGraph(tasks=()))

    assert state.status is RunStatus.COMPLETED
    assert not [n for n in worker.started if task_id_of(n) is not None]


# --- downstream consumers see runtime-admitted work ------------------------


async def test_every_admitted_task_s_code_is_visible_to_downstream_agents(
    tmp_path: Path,
) -> None:
    """`agents/docs.py`, `reviewer.py` and `tester.py` each used to carry
    their own `("impl_domain", "impl_api")` - two of the three nodes that
    existed, so infrastructure code was invisible to all three in every run
    this system has ever performed."""
    state, _, _ = await _run_stage(tmp_path)

    seen = implementation_node_ids(state)
    for task_id in ("domain", "application", "infrastructure", "api"):
        assert impl_node_id(task_id) in seen


async def test_a_build_node_resolves_its_project_from_its_task(tmp_path: Path) -> None:
    """One `dotnet_build` handler serves every build node; it finds the
    project to compile through the task, never by matching a suffix against
    a project name."""
    state, _, _ = await _run_stage(tmp_path)

    task = task_of(state, build_node_id("application"))
    assert task is not None
    assert task.component == "Shop.Application"
    assert task_of(state, "impl_start") is None


def test_edges_into_a_task_carry_no_condition_other_than_the_repair_pair() -> None:
    """Dependency and build edges are `on_success`; only the repair pair
    differs, and its return edge is `on_success` too - never `always`, which
    would let a repair that produced nothing re-trigger a build."""
    proposal = TaskGraphSubgraphProvider().propose(
        _decompose_node(), _state_with(FOUR_LAYER_TASKS, FOUR_LAYER_SKELETON)
    )
    assert proposal is not None
    by_condition: dict[EdgeCondition, set[tuple[str, str]]] = {}
    for edge in proposal.edges:
        by_condition.setdefault(edge.condition, set()).add((edge.source, edge.target))

    assert set(by_condition) == {EdgeCondition.ON_SUCCESS, EdgeCondition.ON_FAILURE}
    assert all(
        source.startswith("build:") and target.startswith("repair:")
        for source, target in by_condition[EdgeCondition.ON_FAILURE]
    )
    assert (repair_node_id("api"), build_node_id("api")) in by_condition[EdgeCondition.ON_SUCCESS]


# --- decompose is recoverable -----------------------------------------------
#
# `decompose` is the one agent whose output the kernel must accept before any
# implementation node exists at all. It used to have no ON_FAILURE edge, so a
# plan that could not be admitted ended the entire run - after gate1 and gate2
# had already been granted. These pin the bounded self-cycle that replaced
# that, and the safe-stop at the end of it.


class _FailsThenSucceedsDecompose:
    """First attempt proposes a plan that leaves a scaffolded project with no
    task - live run `009ea59f-...`'s exact mistake. Second attempt fixes it."""

    def __init__(self) -> None:
        self.attempts = 0
        self.prior_errors: list[str | None] = []

    async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome:
        self.attempts += 1
        self.prior_errors.append(state.node(node.id).last_error)
        if self.attempts == 1:
            return NodeExecutionOutcome(
                ok=False,
                failure_kind=FailureKind.AGENT_PROTOCOL_FAILURE,
                error="no implementation task targets ['Shop.Application']",
            )
        return NodeExecutionOutcome(
            ok=True,
            artifact_kind="TaskGraph",
            artifact_payload=FOUR_LAYER_TASKS.model_dump(mode="json"),
        )


class _AlwaysFailsDecompose:
    def __init__(self) -> None:
        self.attempts = 0

    async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome:
        self.attempts += 1
        return NodeExecutionOutcome(
            ok=False,
            failure_kind=FailureKind.AGENT_PROTOCOL_FAILURE,
            error="no implementation task targets ['Shop.Application']",
        )


def _recoverable_stage_graph() -> WorkflowGraph:
    """`greenfield.yaml`'s decompose node exactly: `join: any` over a
    permanently-succeeded `scaffold` plus its own `on_failure` self-edge,
    bounded by `cycle_budget`."""
    graph = _execution_stage_graph()
    nodes = tuple(
        n.model_copy(update={"join": JoinPolicy.ANY, "cycle_budget": 2})
        if n.id == "decompose"
        else n
        for n in graph.nodes
    )
    edges = (
        *graph.edges,
        Edge(source="decompose", target="decompose", condition=EdgeCondition.ON_FAILURE),
    )
    return WorkflowGraph(name=graph.name, nodes=nodes, edges=edges, entry=graph.entry)


async def _run_recoverable(
    tmp_path: Path, decompose: NodeExecutor
) -> tuple[RunState, _RecordingExecutor, JsonlEventStore]:
    worker = _RecordingExecutor()
    executors: dict[str, NodeExecutor] = {
        "scaffold": _SkeletonExecutor(),
        "decompose": decompose,
        "implementer": worker,
        "dotnet_build": worker,
    }
    store = JsonlEventStore(tmp_path / "events", fsync=False)
    state = await Scheduler(
        _recoverable_stage_graph(),
        store,
        executors,
        entry_gate=BudgetEntryGate(GENEROUS),
        subgraphs=TaskGraphSubgraphProvider(),
    ).run(uuid4())
    return state, worker, store


async def test_a_failed_decompose_is_retried_and_the_run_recovers(tmp_path: Path) -> None:
    decompose = _FailsThenSucceedsDecompose()

    state, _, _ = await _run_recoverable(tmp_path, decompose)

    assert decompose.attempts == 2
    assert state.status is RunStatus.COMPLETED
    # The retry admitted the corrected plan, so implementation actually ran.
    assert state.nodes[impl_node_id("application")].status is NodeStatus.SUCCEEDED


async def test_the_retry_can_see_why_the_previous_attempt_failed(tmp_path: Path) -> None:
    """The feedback channel the agent reads through
    `ContextRetriever.last_error_of`. Without it the second attempt would
    regenerate the same plan and spend the budget for nothing."""
    decompose = _FailsThenSucceedsDecompose()

    await _run_recoverable(tmp_path, decompose)

    assert decompose.prior_errors[0] is None
    assert decompose.prior_errors[1] is not None
    assert "Shop.Application" in decompose.prior_errors[1]


async def test_the_retry_is_visible_in_the_event_log(tmp_path: Path) -> None:
    """A retry inside the agent would have been invisible here. The kernel
    owning the cycle is what makes it auditable."""
    state, _, store = await _run_recoverable(tmp_path, _FailsThenSucceedsDecompose())
    events = [e async for e in store.read(state.run_id)]

    decompose_events = [(e.type, e.attempt) for e in events if e.node_id == "decompose"]
    assert (EventType.NODE_FAILED, 1) in decompose_events
    assert (EventType.NODE_RETRY_SCHEDULED, None) in decompose_events
    assert (EventType.NODE_STARTED, 2) in decompose_events


async def test_an_unfixable_plan_safe_stops_instead_of_looping(tmp_path: Path) -> None:
    """Bounded: the budget is spent, then the run halts. Not RUN_FAILED on
    the first slip, and not an unbounded retry either."""
    decompose = _AlwaysFailsDecompose()

    state, _, _ = await _run_recoverable(tmp_path, decompose)

    assert decompose.attempts == 2  # cycle_budget, not more
    assert state.status is RunStatus.HALTED
    assert "cycle_budget" in (state.halt_reason or "")
    assert state.status_of("impl_start") is NodeStatus.PENDING
