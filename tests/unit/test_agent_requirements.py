"""The `requirements` agent (docs/03 sections 3.1-3.2)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from ases.agents.base import AgentContext
from ases.agents.requirements import RequirementsAgent, RequirementsAgentInput, register_prompts
from ases.context.retriever import ContextRetriever
from ases.contracts.artifacts import RequirementSpec
from ases.kernel.state import ApprovalRecord, RunState
from ases.providers.base import CompletionRequest, CompletionResult
from ases.providers.mock import MockProvider
from ases.providers.prompts.registry import PromptRegistry
from ases.providers.router import ModelRouter


class _RecordingMockProvider(MockProvider):
    """Records every rendered prompt, so a test can assert on prompt
    *content* (e.g. a rejection reason actually reaching the model), not just
    the parsed result."""

    def __init__(self) -> None:
        super().__init__()
        self.rendered_prompts: list[str] = []

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.rendered_prompts.append(request.rendered_prompt)
        return await super().complete(request)


def _rejected_gate1_state() -> RunState:
    state = RunState(run_id=uuid4())
    state.approvals["gate1"] = ApprovalRecord(
        node_id="gate1",
        artifact_hash="h",
        granted=False,
        actor="human:alice",
        decided_at=datetime.now(UTC),
        reason="the summary misses the redirect requirement",
    )
    return state


def _ctx(
    provider: MockProvider,
    *,
    requirement_text: str | None = "build a url shortener",
    state: RunState | None = None,
) -> AgentContext:
    prompts = PromptRegistry()
    register_prompts(prompts)
    return AgentContext(
        run_id="run-1",
        node_id="req",
        retriever=ContextRetriever(state or RunState(run_id=uuid4())),
        provider=provider,
        router=ModelRouter(),
        prompts=prompts,
        run_input={"requirement_text": requirement_text} if requirement_text else {},
    )


def test_build_input_reads_the_requirement_text_from_run_input() -> None:
    agent = RequirementsAgent()
    ctx = _ctx(MockProvider(), requirement_text="build a thing")
    inp = agent.build_input(ctx)
    assert inp == RequirementsAgentInput(requirement_text="build a thing")


def test_build_input_without_requirement_text_raises() -> None:
    agent = RequirementsAgent()
    ctx = _ctx(MockProvider(), requirement_text=None)
    with pytest.raises(ValueError, match="requirement_text"):
        agent.build_input(ctx)


async def test_run_returns_the_parsed_requirement_spec() -> None:
    spec = RequirementSpec(
        summary="a url shortener", in_scope=("create", "redirect"), source_text="raw"
    )
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=spec.model_dump_json(), parsed=spec, model_id="m", stop_reason="end_turn"
        )
    )
    ctx = _ctx(provider)
    agent = RequirementsAgent()

    result = await agent.run(ctx, agent.build_input(ctx))

    assert result.artifact == spec
    assert result.confidence > 0
    assert "requirements.analyze" in result.rationale


async def test_run_registers_at_least_one_citation() -> None:
    spec = RequirementSpec(summary="s", source_text="raw")
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=spec.model_dump_json(), parsed=spec, model_id="m", stop_reason="end_turn"
        )
    )
    ctx = _ctx(provider)
    agent = RequirementsAgent()

    result = await agent.run(ctx, agent.build_input(ctx))
    assert len(result.citations) >= 1


def test_capability_manifest_grants_no_tools() -> None:
    """The requirements agent is pure text analysis - it should need no
    registered tool at all, matching deny-by-default (a manifest granting
    nothing is the correct default, not an oversight)."""
    assert RequirementsAgent.capabilities.allowed_tools == frozenset()


def test_model_needs_ask_for_structured_output() -> None:
    assert RequirementsAgent.model_needs.structured_output is True


def test_build_input_reads_the_gate1_rejection_reason() -> None:
    agent = RequirementsAgent()
    ctx = _ctx(MockProvider(), state=_rejected_gate1_state())

    inp = agent.build_input(ctx)

    assert inp.prior_rejection == "the summary misses the redirect requirement"


def test_build_input_with_no_rejection_leaves_prior_rejection_none() -> None:
    agent = RequirementsAgent()
    ctx = _ctx(MockProvider())

    inp = agent.build_input(ctx)

    assert inp.prior_rejection is None


async def test_run_feeds_the_rejection_reason_into_the_rendered_prompt() -> None:
    spec = RequirementSpec(summary="s", source_text="raw")
    provider = _RecordingMockProvider()
    provider.respond_with(
        CompletionResult(
            text=spec.model_dump_json(), parsed=spec, model_id="m", stop_reason="end_turn"
        )
    )
    ctx = _ctx(provider, state=_rejected_gate1_state())
    agent = RequirementsAgent()

    await agent.run(ctx, agent.build_input(ctx))

    assert len(provider.rendered_prompts) == 1
    assert "the summary misses the redirect requirement" in provider.rendered_prompts[0]


async def test_run_with_no_rejection_never_mentions_one_in_the_prompt() -> None:
    spec = RequirementSpec(summary="s", source_text="raw")
    provider = _RecordingMockProvider()
    provider.respond_with(
        CompletionResult(
            text=spec.model_dump_json(), parsed=spec, model_id="m", stop_reason="end_turn"
        )
    )
    ctx = _ctx(provider)
    agent = RequirementsAgent()

    await agent.run(ctx, agent.build_input(ctx))

    assert "REJECTION_REASON" not in provider.rendered_prompts[0]
