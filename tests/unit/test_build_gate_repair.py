"""The per-task `dotnet build` exit gate's bounded repair cycle (docs/02
Phase 4: "compile failures surface per task, not as a pile-up at
integration"), proven in isolation on a minimal hand-built graph - the same
style `test_greenfield_partial_e2e.py` uses for a slice of the real
`greenfield.yaml` shape rather than the whole file, so a build failure and
its repair pass are pinned down without the concurrency ambiguity of two
implementation tasks sharing one fake `dotnet` runner.

`arch`/`scaffold`/`decompose` are faked since only `impl_domain` ->
`build_domain` -> `repair_domain`'s cycle is under test here; the real
`ImplementerAgent` is used for both `impl_domain` and `repair_domain`,
exactly as `workflows/greenfield.yaml` wires them.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from uuid import uuid4

from ases.agents.executor import AgentNodeExecutor
from ases.agents.implementer import PROMPT_VERSION as IMPLEMENTER_PROMPT_VERSION
from ases.agents.implementer import ImplementerAgent
from ases.agents.implementer import register_prompts as register_implementer_prompts
from ases.agents.tool_executor import ToolNodeExecutor
from ases.context.retriever import ArtifactRef, ContextRetriever
from ases.contracts.artifacts import CodePatch, FileChange
from ases.kernel.events import EventType
from ases.kernel.failures import FailureKind
from ases.kernel.gates import BudgetEntryGate, BudgetLimits
from ases.kernel.graph import Edge, EdgeCondition, JoinPolicy, NodeKind, NodeSpec, WorkflowGraph
from ases.kernel.scheduler import NodeExecutionOutcome, NodeExecutor, Scheduler
from ases.kernel.state import NodeStatus, RunState, RunStatus
from ases.kernel.store.jsonl import JsonlEventStore
from ases.kernel.tools.dotnet import build_dotnet_tools
from ases.kernel.tools.fs import WRITE_FILE
from ases.kernel.tools.registry import ToolRegistry
from ases.providers.base import CompletionResult
from ases.providers.mock import MockProvider
from ases.providers.prompts.registry import PromptRegistry
from ases.providers.router import ModelRouter

GENEROUS = BudgetLimits(max_tokens=10_000_000, max_usd=1_000.0, max_wallclock_seconds=3600)


class _FirstBuildFailsRunner:
    """The first `dotnet build` call fails with a compiler diagnostic; every
    later call (the post-repair retry) succeeds."""

    def __init__(self) -> None:
        self.calls: list[Sequence[str]] = []
        self._build_calls = 0

    async def __call__(self, argv: Sequence[str], cwd: Path) -> tuple[int, str, str]:
        self.calls.append(tuple(argv))
        if tuple(argv[:2]) == ("dotnet", "build"):
            self._build_calls += 1
            if self._build_calls == 1:
                return 1, "", "Program.cs(12,9): error CS0103: 'Foo' does not exist"
        return 0, "", ""


class _FixedArtifactExecutor:
    """Always produces the same artifact kind/payload - the fake counterpart
    to a real agent for a node not under test here."""

    def __init__(self, kind: str, payload: dict[str, object]) -> None:
        self._kind = kind
        self._payload = payload

    async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome:
        return NodeExecutionOutcome(
            ok=True, artifact_kind=self._kind, artifact_payload=self._payload
        )


def _graph() -> WorkflowGraph:
    return WorkflowGraph(
        name="build_gate_slice",
        entry=("req",),
        nodes=(
            NodeSpec(id="req", kind=NodeKind.AGENT, handler="fake_req", produces="RequirementSpec"),
            NodeSpec(id="arch", kind=NodeKind.AGENT, handler="fake_arch", produces="DesignSpec"),
            NodeSpec(
                id="scaffold",
                kind=NodeKind.AGENT,
                handler="fake_scaffold",
                produces="SolutionSkeleton",
            ),
            NodeSpec(
                id="decompose", kind=NodeKind.AGENT, handler="fake_decompose", produces="TaskGraph"
            ),
            NodeSpec(
                id="impl_domain", kind=NodeKind.AGENT, handler="implementer", produces="CodePatch"
            ),
            NodeSpec(
                id="build_domain",
                kind=NodeKind.TOOL,
                handler="dotnet_build",
                join=JoinPolicy.ANY,
                cycle_budget=2,
            ),
            NodeSpec(
                id="repair_domain", kind=NodeKind.AGENT, handler="implementer", produces="CodePatch"
            ),
            NodeSpec(id="done", kind=NodeKind.TERMINAL),
        ),
        edges=(
            Edge(source="req", target="arch"),
            Edge(source="arch", target="scaffold"),
            Edge(source="scaffold", target="decompose"),
            Edge(source="decompose", target="impl_domain"),
            Edge(source="impl_domain", target="build_domain"),
            Edge(source="build_domain", target="repair_domain", condition=EdgeCondition.ON_FAILURE),
            Edge(source="repair_domain", target="build_domain", condition=EdgeCondition.ALWAYS),
            Edge(source="build_domain", target="done"),
        ),
    )


async def test_a_failed_per_task_build_triggers_repair_and_then_succeeds(
    tmp_path: Path,
) -> None:
    prompts = PromptRegistry()
    register_implementer_prompts(prompts)

    first_patch = CodePatch(
        summary="initial attempt",
        files=(FileChange(path="Domain/ShortUrl.cs", content="public class ShortUrl { Foo }"),),
    )
    fixed_patch = CodePatch(
        summary="fixed the compile error",
        files=(FileChange(path="Domain/ShortUrl.cs", content="public class ShortUrl { }"),),
    )
    provider = MockProvider()
    provider.respond_to(
        f"implementer.write@v{IMPLEMENTER_PROMPT_VERSION}",
        CompletionResult(
            text=first_patch.model_dump_json(),
            parsed=first_patch,
            model_id="m",
            stop_reason="end_turn",
        ),
    )
    provider.respond_to(
        f"implementer.write@v{IMPLEMENTER_PROMPT_VERSION}",
        CompletionResult(
            text=fixed_patch.model_dump_json(),
            parsed=fixed_patch,
            model_id="m",
            stop_reason="end_turn",
        ),
    )

    tools = ToolRegistry()
    tools.register(WRITE_FILE)
    runner = _FirstBuildFailsRunner()
    for spec in build_dotnet_tools(runner):
        tools.register(spec)
    tool_cwd = PurePosixPath(tmp_path.as_posix())

    implementer_executor = AgentNodeExecutor(
        ImplementerAgent(),
        provider=provider,
        router=ModelRouter(),
        prompts=prompts,
        tools=tools,
        tool_cwd=tool_cwd,
    )

    executors: dict[str, NodeExecutor] = {
        # Payloads deliberately distinct, not `{}` - content addressing hashes
        # only the payload, not the kind (`_finish_agent_node`), so two empty
        # artifacts of different kinds would otherwise collide on one hash.
        "fake_req": _FixedArtifactExecutor(
            "RequirementSpec", {"summary": "r", "source_text": "build a thing"}
        ),
        "fake_arch": _FixedArtifactExecutor("DesignSpec", {"summary": "d"}),
        "fake_scaffold": _FixedArtifactExecutor(
            "SolutionSkeleton", {"projects": ["UrlShortener.Domain"]}
        ),
        "fake_decompose": _FixedArtifactExecutor("TaskGraph", {"tasks": []}),
        "implementer": implementer_executor,
        "dotnet_build": ToolNodeExecutor("dotnet.build", tools=tools, tool_cwd=tool_cwd),
    }

    store = JsonlEventStore(tmp_path / "events")
    scheduler = Scheduler(_graph(), store, executors, entry_gate=BudgetEntryGate(GENEROUS))

    state = await scheduler.run(uuid4())

    assert state.status is RunStatus.COMPLETED
    assert state.nodes["impl_domain"].status is NodeStatus.SUCCEEDED
    assert state.nodes["build_domain"].status is NodeStatus.SUCCEEDED
    assert state.nodes["repair_domain"].status is NodeStatus.SUCCEEDED

    build_calls = [c for c in runner.calls if tuple(c[:2]) == ("dotnet", "build")]
    assert len(build_calls) == 2  # the failing attempt, then the post-repair retry

    # repair_domain genuinely read build_domain's own failure, not a fixture.
    retriever = ContextRetriever(state)
    repair_hash = state.nodes["repair_domain"].produced[0]
    fetched = retriever.fetch(ArtifactRef(artifact_hash=repair_hash))
    assert isinstance(fetched, CodePatch)
    assert fetched.summary == "fixed the compile error"


# --- what a *failed* build is allowed to do --------------------------------


class _AlwaysFailingExecutor:
    """A build that never goes green, however often it is retried."""

    async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome:
        return NodeExecutionOutcome(
            ok=False,
            failure_kind=FailureKind.BUILD_FAILURE,
            error="Program.cs(1,1): error CS0246: type or namespace not found",
        )


class _AlwaysOkExecutor:
    async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome:
        return NodeExecutionOutcome(ok=True)


def _two_builds_into_a_barrier() -> WorkflowGraph:
    """`workflows/greenfield.yaml`'s exact shapes: a per-task build gate with
    an `on_failure` edge out to a repair position and an edge back in, two
    such builds converging on a `join: all` barrier, and - the detail under
    test - the build-to-barrier edges written with **no condition at all**,
    exactly as the YAML writes them."""
    return WorkflowGraph(
        name="failed_build_barrier",
        entry=("start",),
        nodes=(
            NodeSpec(id="start", kind=NodeKind.AGENT, handler="ok"),
            NodeSpec(id="build_ok", kind=NodeKind.TOOL, handler="ok"),
            NodeSpec(
                id="build_bad",
                kind=NodeKind.TOOL,
                handler="bad",
                join=JoinPolicy.ANY,
                cycle_budget=2,
            ),
            NodeSpec(id="repair_bad", kind=NodeKind.AGENT, handler="ok"),
            NodeSpec(id="barrier", kind=NodeKind.BARRIER, join=JoinPolicy.ALL),
            NodeSpec(id="done", kind=NodeKind.TERMINAL),
        ),
        edges=(
            Edge(source="start", target="build_ok"),
            Edge(source="start", target="build_bad"),
            Edge(source="build_bad", target="repair_bad", condition=EdgeCondition.ON_FAILURE),
            Edge(source="repair_bad", target="build_bad", condition=EdgeCondition.ON_SUCCESS),
            Edge(source="build_ok", target="barrier"),
            Edge(source="build_bad", target="barrier"),
            Edge(source="barrier", target="done"),
        ),
    )


async def test_a_failed_build_never_satisfies_the_barrier(tmp_path: Path) -> None:
    """The property `workflows/greenfield.yaml` depends on and never stated:
    a build that never succeeds cannot let the run past the barrier, no
    matter how many repair attempts it consumes or how many *sibling* builds
    went green. Pinned here so a future edit to the YAML's edge conditions -
    or to `_join_satisfied` - cannot quietly turn a failed compile into a
    satisfied prerequisite."""
    executors: dict[str, NodeExecutor] = {
        "ok": _AlwaysOkExecutor(),
        "bad": _AlwaysFailingExecutor(),
    }
    store = JsonlEventStore(tmp_path / "events", fsync=False)
    scheduler = Scheduler(
        _two_builds_into_a_barrier(), store, executors, entry_gate=BudgetEntryGate(GENEROUS)
    )

    state = await scheduler.run(uuid4())

    assert state.nodes["build_ok"].status is NodeStatus.SUCCEEDED
    assert state.nodes["build_bad"].status is not NodeStatus.SUCCEEDED
    assert state.nodes["build_bad"].last_error is not None
    # Never even staged: the barrier has no state entry at all, so nothing
    # downstream of it could have run.
    assert "barrier" not in state.nodes
    assert state.status_of("barrier") is NodeStatus.PENDING
    assert state.status_of("done") is NodeStatus.PENDING
    assert state.status is RunStatus.HALTED
    assert "cycle_budget" in (state.halt_reason or "")


def test_an_edge_written_without_a_condition_is_an_on_success_edge() -> None:
    """`- { source: build_domain, target: barrier }` in the workflow YAML
    carries no `condition:` key. This is what it means."""
    assert Edge(source="a", target="b").condition is EdgeCondition.ON_SUCCESS
    assert EdgeCondition.ON_SUCCESS.matches(NodeStatus.SUCCEEDED) is True
    assert EdgeCondition.ON_SUCCESS.matches(NodeStatus.FAILED) is False
    # `always` is the union of every settled outcome, failure included -
    # which is why the repair-to-build edges are no longer written with it.
    assert EdgeCondition.ALWAYS.matches(NodeStatus.FAILED) is True


async def test_a_repair_that_produced_nothing_does_not_re_trigger_its_build(
    tmp_path: Path,
) -> None:
    """The live-run failure mode `on_success` on the repair-to-build edge
    removes. `repair_bad` fails at the LLM boundary (an agent protocol
    failure, not a source-code failure); under the old `always` edge that
    still re-dispatched `build_bad`, burning its `cycle_budget` on a rebuild
    of unchanged source and halting with the *build's* budget named as the
    reason. The run must now stop at the repair and report what actually
    went wrong."""
    executors: dict[str, NodeExecutor] = {
        "ok": _AlwaysOkExecutor(),
        "bad": _AlwaysFailingExecutor(),
        "broken_repair": _ProtocolFailureExecutor(),
    }
    graph = _two_builds_into_a_barrier()
    graph = WorkflowGraph(
        name=graph.name,
        nodes=tuple(
            n.model_copy(update={"handler": "broken_repair"}) if n.id == "repair_bad" else n
            for n in graph.nodes
        ),
        edges=graph.edges,
        entry=graph.entry,
    )
    store = JsonlEventStore(tmp_path / "events", fsync=False)

    state = await Scheduler(graph, store, executors, entry_gate=BudgetEntryGate(GENEROUS)).run(
        uuid4()
    )

    assert state.status is RunStatus.FAILED
    # `build_bad` ran exactly once - it was never re-dispatched by a repair
    # that had nothing to offer it.
    assert state.nodes["build_bad"].attempt == 1
    assert state.nodes["repair_bad"].status is NodeStatus.FAILED
    events = [e async for e in store.read(state.run_id)]
    failed = [e for e in events if e.type is EventType.NODE_FAILED and e.node_id == "repair_bad"]
    assert failed[-1].payload["failure_kind"] == FailureKind.AGENT_PROTOCOL_FAILURE.value


class _ProtocolFailureExecutor:
    """An agent whose model returned output that could not be used at all."""

    async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome:
        return NodeExecutionOutcome(
            ok=False,
            failure_kind=FailureKind.AGENT_PROTOCOL_FAILURE,
            error="CodePatch: the model's response was truncated at max_tokens",
        )
