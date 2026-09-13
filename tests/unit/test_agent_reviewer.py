"""The `reviewer` agent (docs/03 section 3.3's `CODE_REVIEW` node - advisory,
per docs/05 section 3.7)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from ases.agents.base import AgentContext
from ases.agents.reviewer import ReviewerAgent, register_prompts
from ases.context.retriever import ContextRetriever
from ases.contracts.artifacts import CodePatch, DesignSpec, Finding, ReviewReport, TestSuite
from ases.kernel.state import ArtifactRecord, NodeState, RunState
from ases.providers.base import CompletionResult
from ases.providers.mock import MockProvider
from ases.providers.prompts.registry import PromptRegistry
from ases.providers.router import ModelRouter


def _record(hash_: str, kind: str, node_id: str) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_hash=hash_, kind=kind, node_id=node_id, produced_at=datetime.now(UTC)
    )


def _state() -> RunState:
    state = RunState(run_id=uuid4())
    design = DesignSpec(summary="d")
    domain = CodePatch(summary="domain entity")
    api = CodePatch(summary="api controller")
    tests = TestSuite(summary="unit tests")

    state.artifacts["h-design"] = _record("h-design", "DesignSpec", "arch")
    state.artifact_content["h-design"] = design.model_dump(mode="json")
    state.nodes["arch"] = NodeState(node_id="arch", produced=("h-design",))

    state.artifacts["h-domain"] = _record("h-domain", "CodePatch", "impl_domain")
    state.artifact_content["h-domain"] = domain.model_dump(mode="json")
    state.nodes["impl_domain"] = NodeState(node_id="impl_domain", produced=("h-domain",))

    state.artifacts["h-api"] = _record("h-api", "CodePatch", "impl_api")
    state.artifact_content["h-api"] = api.model_dump(mode="json")
    state.nodes["impl_api"] = NodeState(node_id="impl_api", produced=("h-api",))

    state.artifacts["h-tests"] = _record("h-tests", "TestSuite", "test_gen")
    state.artifact_content["h-tests"] = tests.model_dump(mode="json")
    state.nodes["test_gen"] = NodeState(node_id="test_gen", produced=("h-tests",))
    return state


def _ctx(provider: MockProvider, state: RunState) -> AgentContext:
    prompts = PromptRegistry()
    register_prompts(prompts)
    return AgentContext(
        run_id="run-1",
        node_id="review",
        retriever=ContextRetriever(state),
        provider=provider,
        router=ModelRouter(),
        prompts=prompts,
    )


def test_build_input_reads_implementations_and_tests() -> None:
    ctx = _ctx(MockProvider(), _state())
    inp = ReviewerAgent().build_input(ctx)
    assert len(inp.implementations) == 2
    assert inp.tests.summary == "unit tests"


async def test_run_returns_the_review_report() -> None:
    report = ReviewReport(
        findings=(Finding(summary="missing null check", citation="A.cs:12", severity="warning"),),
        verdict="concerns",
    )
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=report.model_dump_json(), parsed=report, model_id="m", stop_reason="end_turn"
        )
    )
    ctx = _ctx(provider, _state())
    agent = ReviewerAgent()

    result = await agent.run(ctx, agent.build_input(ctx))

    assert result.artifact == report
    assert "advisory" in result.rationale.lower()


def test_capability_manifest_grants_no_tools() -> None:
    """The reviewer only reads and reasons - it never touches a file."""
    assert ReviewerAgent.capabilities.allowed_tools == frozenset()
