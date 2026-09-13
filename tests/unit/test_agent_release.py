"""The `release` agent (docs/03 section 3.3's `RELEASE_READINESS` node) -
the first agent with a genuinely optional upstream artifact (`sec_scan`)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from ases.agents.base import AgentContext
from ases.agents.release import ReleaseAgent, register_prompts
from ases.context.retriever import ContextRetriever
from ases.contracts.artifacts import (
    DocsPatch,
    PolicyViolation,
    ReleaseReport,
    ReviewReport,
)
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


def _record(hash_: str, kind: str, node_id: str) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_hash=hash_, kind=kind, node_id=node_id, produced_at=datetime.now(UTC)
    )


def _state(*, with_security_findings: bool) -> RunState:
    state = RunState(run_id=uuid4())
    review = ReviewReport(verdict="pass")
    docs = DocsPatch(summary="docs written")

    state.artifacts["h-review"] = _record("h-review", "ReviewReport", "review")
    state.artifact_content["h-review"] = review.model_dump(mode="json")
    state.nodes["review"] = NodeState(node_id="review", produced=("h-review",))

    state.artifacts["h-docs"] = _record("h-docs", "DocsPatch", "docs_gen")
    state.artifact_content["h-docs"] = docs.model_dump(mode="json")
    state.nodes["docs_gen"] = NodeState(node_id="docs_gen", produced=("h-docs",))

    if with_security_findings:
        violation = PolicyViolation(rule="secrets_detected", message="an AWS key was found")
        state.artifacts["h-sec"] = _record("h-sec", "PolicyViolation", "sec_scan")
        state.artifact_content["h-sec"] = violation.model_dump(mode="json")
        state.nodes["sec_scan"] = NodeState(node_id="sec_scan", produced=("h-sec",))
    return state


def _ctx(provider: MockProvider, state: RunState) -> AgentContext:
    prompts = PromptRegistry()
    register_prompts(prompts)
    return AgentContext(
        run_id="run-1",
        node_id="release",
        retriever=ContextRetriever(state),
        provider=provider,
        router=ModelRouter(),
        prompts=prompts,
    )


def test_build_input_with_no_security_findings_leaves_it_none() -> None:
    ctx = _ctx(MockProvider(), _state(with_security_findings=False))
    inp = ReleaseAgent().build_input(ctx)
    assert inp.security_findings is None
    assert inp.review.verdict == "pass"


def test_build_input_with_security_findings_reads_them() -> None:
    ctx = _ctx(MockProvider(), _state(with_security_findings=True))
    inp = ReleaseAgent().build_input(ctx)
    assert inp.security_findings is not None
    assert inp.security_findings.rule == "secrets_detected"


async def test_run_on_a_clean_scan_does_not_cite_sec_scan() -> None:
    report = ReleaseReport(ready=True, checklist=("build passed", "tests passed"))
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=report.model_dump_json(), parsed=report, model_id="m", stop_reason="end_turn"
        )
    )
    ctx = _ctx(provider, _state(with_security_findings=False))
    agent = ReleaseAgent()

    result = await agent.run(ctx, agent.build_input(ctx))

    assert result.artifact == report
    assert all("sec_scan" not in c.source for c in result.citations)


async def test_run_on_findings_cites_sec_scan() -> None:
    report = ReleaseReport(ready=False, risks=("unresolved secret in source",))
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=report.model_dump_json(), parsed=report, model_id="m", stop_reason="end_turn"
        )
    )
    ctx = _ctx(provider, _state(with_security_findings=True))
    agent = ReleaseAgent()

    result = await agent.run(ctx, agent.build_input(ctx))

    assert any("sec_scan" in c.source for c in result.citations)


def test_capability_manifest_grants_no_tools() -> None:
    assert ReleaseAgent.capabilities.allowed_tools == frozenset()


def _state_with_gate3_rejection() -> RunState:
    state = _state(with_security_findings=False)
    state.approvals["gate3"] = ApprovalRecord(
        node_id="gate3",
        artifact_hash="h-release",
        granted=False,
        actor="human:alice",
        decided_at=datetime.now(UTC),
        reason="checklist is too thin",
    )
    return state


def test_build_input_reads_the_gate3_rejection_reason() -> None:
    ctx = _ctx(MockProvider(), _state_with_gate3_rejection())
    inp = ReleaseAgent().build_input(ctx)
    assert inp.prior_rejection == "checklist is too thin"


def test_build_input_with_no_rejection_leaves_prior_rejection_none() -> None:
    ctx = _ctx(MockProvider(), _state(with_security_findings=False))
    inp = ReleaseAgent().build_input(ctx)
    assert inp.prior_rejection is None


async def test_run_feeds_the_rejection_reason_into_the_rendered_prompt() -> None:
    report = ReleaseReport(ready=True)
    provider = _RecordingMockProvider()
    provider.respond_with(
        CompletionResult(
            text=report.model_dump_json(), parsed=report, model_id="m", stop_reason="end_turn"
        )
    )
    ctx = _ctx(provider, _state_with_gate3_rejection())
    agent = ReleaseAgent()

    await agent.run(ctx, agent.build_input(ctx))

    assert "checklist is too thin" in provider.rendered_prompts[0]
