"""The `decompose` agent (docs/03 section 3.3's `DECOMPOSE` node)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from ases.agents.base import AgentContext
from ases.agents.decompose import DecomposeAgent, DecomposeAgentInput, register_prompts
from ases.context.retriever import ContextRetriever, NoArtifactFromNodeError
from ases.contracts.artifacts import DesignSpec, SolutionSkeleton, TaskGraph, TaskSpec
from ases.kernel.state import ArtifactRecord, NodeState, RunState
from ases.providers.base import CompletionResult
from ases.providers.mock import MockProvider
from ases.providers.prompts.registry import PromptRegistry
from ases.providers.router import ModelRouter


def _record(hash_: str, kind: str, node_id: str) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_hash=hash_, kind=kind, node_id=node_id, produced_at=datetime.now(UTC)
    )


def _state_with(design: DesignSpec, skeleton: SolutionSkeleton) -> RunState:
    state = RunState(run_id=uuid4())
    state.artifacts["h-design"] = _record("h-design", "DesignSpec", "arch")
    state.artifact_content["h-design"] = design.model_dump(mode="json")
    state.nodes["arch"] = NodeState(node_id="arch", produced=("h-design",))
    state.artifacts["h-skel"] = _record("h-skel", "SolutionSkeleton", "scaffold")
    state.artifact_content["h-skel"] = skeleton.model_dump(mode="json")
    state.nodes["scaffold"] = NodeState(node_id="scaffold", produced=("h-skel",))
    return state


def _ctx(provider: MockProvider, state: RunState) -> AgentContext:
    prompts = PromptRegistry()
    register_prompts(prompts)
    return AgentContext(
        run_id="run-1",
        node_id="decompose",
        retriever=ContextRetriever(state),
        provider=provider,
        router=ModelRouter(),
        prompts=prompts,
    )


def test_build_input_reads_design_and_skeleton() -> None:
    design = DesignSpec(summary="d")
    skeleton = SolutionSkeleton(projects=("A",))
    ctx = _ctx(MockProvider(), _state_with(design, skeleton))
    agent = DecomposeAgent()

    inp = agent.build_input(ctx)

    assert inp == DecomposeAgentInput(design=design, skeleton=skeleton)


def test_build_input_without_a_skeleton_raises() -> None:
    ctx = _ctx(MockProvider(), RunState(run_id=uuid4()))
    with pytest.raises(NoArtifactFromNodeError):
        DecomposeAgent().build_input(ctx)


async def test_run_returns_the_task_graph() -> None:
    design = DesignSpec(summary="d")
    skeleton = SolutionSkeleton(projects=("A", "B"))
    graph = TaskGraph(
        tasks=(
            TaskSpec(id="t1", description="domain entity"),
            TaskSpec(id="t2", description="api controller", depends_on=("t1",)),
        )
    )
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=graph.model_dump_json(), parsed=graph, model_id="m", stop_reason="end_turn"
        )
    )
    ctx = _ctx(provider, _state_with(design, skeleton))
    agent = DecomposeAgent()

    result = await agent.run(ctx, agent.build_input(ctx))

    assert result.artifact == graph
    assert len(result.citations) == 2


def test_capability_manifest_grants_no_tools() -> None:
    assert DecomposeAgent.capabilities.allowed_tools == frozenset()
