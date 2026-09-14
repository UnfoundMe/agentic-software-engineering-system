"""The `decompose` agent (docs/03 section 3.3's `DECOMPOSE` node)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from ases.agents.base import AgentContext
from ases.agents.decompose import (
    DecomposeAgent,
    DecomposeAgentInput,
    DecomposePlanError,
    register_prompts,
)
from ases.context.retriever import ContextRetriever, NoArtifactFromNodeError
from ases.contracts.artifacts import DesignSpec, SolutionSkeleton, TaskGraph, TaskSpec
from ases.kernel.state import ArtifactRecord, NodeState, RunState
from ases.providers.base import CompletionRequest, CompletionResult
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
            TaskSpec(id="t1", description="domain entity", component="A"),
            TaskSpec(id="t2", description="api controller", component="B", depends_on=("t1",)),
        )
    )
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=graph.model_dump_json(), parsed=graph, model_id="m", stop_reason="end_turn"
        )
    )
    recording = _Recording(provider)
    ctx = _ctx(recording, _state_with(design, skeleton))  # type: ignore[arg-type]
    agent = DecomposeAgent()

    result = await agent.run(ctx, agent.build_input(ctx))

    assert result.artifact == graph
    assert len(result.citations) == 2


def test_capability_manifest_grants_no_tools() -> None:
    assert DecomposeAgent.capabilities.allowed_tools == frozenset()


# --- the plan has to fit the solution it plans for -------------------------


class _Recording:
    """Wraps `MockProvider` and keeps every request, so a test can assert what
    the *retry* was actually told - `MockProvider` itself records nothing."""

    def __init__(self, inner: MockProvider) -> None:
        self._inner = inner
        self.requests: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.requests.append(request)
        return await self._inner.complete(request)


async def test_a_plan_that_leaves_a_project_unimplemented_fails_the_node() -> None:
    """The agent does not retry itself - `workflows/greenfield.yaml` gives
    `decompose` an `on_failure` self-edge and the kernel re-dispatches it.
    What this agent owes is a clean, specific failure naming the project that
    was left out, so the retry has something to act on."""
    design = DesignSpec(summary="d")
    skeleton = SolutionSkeleton(projects=("A", "B"))
    incomplete = TaskGraph(tasks=(TaskSpec(id="t1", description="domain", component="A"),))
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=incomplete.model_dump_json(),
            parsed=incomplete,
            model_id="m",
            stop_reason="end_turn",
        )
    )
    recording = _Recording(provider)
    ctx = _ctx(recording, _state_with(design, skeleton))  # type: ignore[arg-type]
    agent = DecomposeAgent()

    with pytest.raises(DecomposePlanError, match=r"no implementation task targets \['B'\]"):
        await agent.run(ctx, agent.build_input(ctx))

    # Exactly one model call: no hidden loop inside the agent.
    assert len(recording.requests) == 1


async def test_a_retry_is_told_why_the_previous_plan_was_rejected() -> None:
    """The kernel re-dispatches the node; `build_input` reads this node's own
    `last_error` and feeds it back, the same way `agents/architect.py`
    responds to a rejected gate. Without this the second attempt would
    regenerate the same plan and burn the cycle budget for nothing - which is
    exactly what `repair_api` did in live run `009ea59f-...`."""
    design = DesignSpec(summary="d")
    skeleton = SolutionSkeleton(projects=("A", "B"))
    complete = TaskGraph(
        tasks=(
            TaskSpec(id="t1", description="domain", component="A"),
            TaskSpec(id="t2", description="api", component="B", depends_on=("t1",)),
        )
    )
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=complete.model_dump_json(), parsed=complete, model_id="m", stop_reason="end_turn"
        )
    )
    state = _state_with(design, skeleton)
    # As the fold leaves it after a failed first attempt.
    state.nodes["decompose"] = NodeState(
        node_id="decompose",
        last_error="no implementation task targets ['B'] - the scaffold created 2 project(s)",
    )
    recording = _Recording(provider)
    ctx = _ctx(recording, state)  # type: ignore[arg-type]
    agent = DecomposeAgent()

    result = await agent.run(ctx, agent.build_input(ctx))

    assert result.artifact == complete
    prompt = recording.requests[0].rendered_prompt
    assert "PLAN_PROBLEM" in prompt
    assert "no implementation task targets ['B']" in prompt


async def test_a_first_attempt_carries_no_correction_section() -> None:
    design = DesignSpec(summary="d")
    skeleton = SolutionSkeleton(projects=("A",))
    plan = TaskGraph(tasks=(TaskSpec(id="t1", description="domain", component="A"),))
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=plan.model_dump_json(), parsed=plan, model_id="m", stop_reason="end_turn"
        )
    )
    recording = _Recording(provider)
    ctx = _ctx(recording, _state_with(design, skeleton))  # type: ignore[arg-type]
    agent = DecomposeAgent()

    await agent.run(ctx, agent.build_input(ctx))

    assert "PLAN_PROBLEM" not in recording.requests[0].rendered_prompt
