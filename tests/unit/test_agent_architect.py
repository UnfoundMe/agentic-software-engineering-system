"""The `architect` agent (docs/03 sections 3.1a, 3.3)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from ases.agents.architect import ArchitectAgent, ArchitectAgentInput, register_prompts
from ases.agents.base import AgentContext
from ases.context.retriever import ContextRetriever, NoArtifactFromNodeError
from ases.contracts.artifacts import DesignSpec, RequirementSpec
from ases.kernel.state import ApprovalRecord, ArtifactRecord, NodeState, RunState
from ases.providers.base import CompletionRequest, CompletionResult
from ases.providers.mock import MockProvider
from ases.providers.prompts.registry import PromptRegistry
from ases.providers.router import ModelRouter


class _RecordingMockProvider(MockProvider):
    def __init__(self) -> None:
        super().__init__()
        self.rendered_prompts: list[str] = []

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.rendered_prompts.append(request.rendered_prompt)
        return await super().complete(request)


def _state_with_requirement(
    spec: RequirementSpec, *, gate2_rejection_reason: str | None = None
) -> RunState:
    state = RunState(run_id=uuid4())
    record = ArtifactRecord(
        artifact_hash="h-req",
        kind="RequirementSpec",
        node_id="req",
        produced_at=datetime.now(UTC),
    )
    state.artifacts["h-req"] = record
    state.artifact_content["h-req"] = spec.model_dump(mode="json")
    state.nodes["req"] = NodeState(node_id="req", produced=("h-req",))
    if gate2_rejection_reason is not None:
        state.approvals["gate2"] = ApprovalRecord(
            node_id="gate2",
            artifact_hash="h-design",
            granted=False,
            actor="human:alice",
            decided_at=datetime.now(UTC),
            reason=gate2_rejection_reason,
        )
    return state


def _ctx(provider: MockProvider, state: RunState) -> AgentContext:
    prompts = PromptRegistry()
    register_prompts(prompts)
    return AgentContext(
        run_id="run-1",
        node_id="arch",
        retriever=ContextRetriever(state),
        provider=provider,
        router=ModelRouter(),
        prompts=prompts,
    )


def test_build_input_reads_the_requirement_from_the_req_node() -> None:
    spec = RequirementSpec(summary="build a url shortener", source_text="raw")
    agent = ArchitectAgent()
    ctx = _ctx(MockProvider(), _state_with_requirement(spec))

    inp = agent.build_input(ctx)

    assert inp == ArchitectAgentInput(requirement=spec)


def test_build_input_without_an_approved_requirement_raises() -> None:
    agent = ArchitectAgent()
    ctx = _ctx(MockProvider(), RunState(run_id=uuid4()))
    with pytest.raises(NoArtifactFromNodeError):
        agent.build_input(ctx)


async def test_run_returns_the_parsed_design_spec() -> None:
    spec = RequirementSpec(summary="build a url shortener", source_text="raw")
    design = DesignSpec(
        summary="4-project layered solution",
        layers=("Domain", "Application", "Infrastructure", "Api"),
        key_decisions=("EF Core over Dapper for velocity",),
    )
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=design.model_dump_json(), parsed=design, model_id="m", stop_reason="end_turn"
        )
    )
    ctx = _ctx(provider, _state_with_requirement(spec))
    agent = ArchitectAgent()

    result = await agent.run(ctx, agent.build_input(ctx))

    assert result.artifact == design
    assert "architect.design" in result.rationale


def test_model_needs_high_reasoning_for_design_work() -> None:
    assert ArchitectAgent.model_needs.reasoning == "high"


def test_capability_manifest_grants_no_tools() -> None:
    assert ArchitectAgent.capabilities.allowed_tools == frozenset()


def test_build_input_reads_the_gate2_rejection_reason() -> None:
    spec = RequirementSpec(summary="s", source_text="raw")
    agent = ArchitectAgent()
    state = _state_with_requirement(spec, gate2_rejection_reason="Domain depends on EF Core")

    inp = agent.build_input(_ctx(MockProvider(), state))

    assert inp.prior_rejection == "Domain depends on EF Core"


def test_build_input_with_no_rejection_leaves_prior_rejection_none() -> None:
    spec = RequirementSpec(summary="s", source_text="raw")
    agent = ArchitectAgent()

    inp = agent.build_input(_ctx(MockProvider(), _state_with_requirement(spec)))

    assert inp.prior_rejection is None


async def test_run_feeds_the_rejection_reason_into_the_rendered_prompt() -> None:
    spec = RequirementSpec(summary="s", source_text="raw")
    design = DesignSpec(summary="d")
    provider = _RecordingMockProvider()
    provider.respond_with(
        CompletionResult(
            text=design.model_dump_json(), parsed=design, model_id="m", stop_reason="end_turn"
        )
    )
    state = _state_with_requirement(spec, gate2_rejection_reason="Domain depends on EF Core")
    agent = ArchitectAgent()
    ctx = _ctx(provider, state)

    await agent.run(ctx, agent.build_input(ctx))

    assert "Domain depends on EF Core" in provider.rendered_prompts[0]
