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
from ases.agents.implementer import ImplementerAgent
from ases.agents.implementer import register_prompts as register_implementer_prompts
from ases.agents.tool_executor import ToolNodeExecutor
from ases.context.retriever import ArtifactRef, ContextRetriever
from ases.contracts.artifacts import CodePatch, FileChange
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
        entry=("arch",),
        nodes=(
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
        "implementer.write@v4",
        CompletionResult(
            text=first_patch.model_dump_json(),
            parsed=first_patch,
            model_id="m",
            stop_reason="end_turn",
        ),
    )
    provider.respond_to(
        "implementer.write@v4",
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
