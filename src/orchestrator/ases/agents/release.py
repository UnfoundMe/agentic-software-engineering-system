"""The `release` agent (docs/03 section 3.3's `RELEASE_READINESS` node;
`workflows/greenfield.yaml`'s `release` node, reached after `quality_gate`
joins `review`, `sec_scan`, and `docs_gen`).

The first agent with a genuinely *optional* upstream artifact:
`kernel/tools/security.py`'s `scan_for_secrets` (behind `sec_scan`) only
produces a `PolicyViolation` when it finds something
(`agents/tool_executor.py`'s `build_artifact` callback returns `None` on a
clean scan) - a clean run legitimately has nothing there. `build_input`
below catches `NoArtifactFromNodeError` for that one upstream only, rather
than treating an empty scan as a build_input failure.

**Gate 3's rejection cycle, and its real limit.** `gate3 -[on_rejected]->
release` re-dispatches this agent (bounded by this node's own
`cycle_budget`), reading the reason via `ContextRetriever.rejection_reason_of`
the same way the other three gated agents do. Disclosed honestly: this agent
only assesses `review`/`sec_scan`/`docs_gen` - none of which change on a
retry - so a redo can only change this agent's own recommendation (weighing
the human's stated concern more heavily, adjusting `risks`/`checklist`), not
the underlying code the concern might actually be about.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from ases.agents.base import Agent, AgentContext, AgentResult, Citation
from ases.context.retriever import NoArtifactFromNodeError
from ases.contracts.artifacts import DocsPatch, PolicyViolation, ReleaseReport, ReviewReport
from ases.kernel.policy import CapabilityManifest
from ases.providers.models import ModelNeeds
from ases.providers.prompts.registry import PromptRegistry, PromptTemplate

PROMPT_NAME = "release.assess"
PROMPT_VERSION = 1
PROMPT_TEMPLATE = """You are the Release Agent in a governed software \
engineering system. You never decide what happens next in the workflow - \
Gate 3 is a human decision; you only assess readiness for that human to \
review.

Review verdict: {review_verdict}
Review findings: {review_findings}
Security scan: {security_summary}
Documentation summary: {docs_summary}

Produce a ReleaseReport: whether you assess the work as ready (a \
recommendation, not a decision), a checklist of what was verified, and any \
risks a human approver should weigh before Gate 3.
{prior_rejection_section}"""


def register_prompts(registry: PromptRegistry) -> None:
    registry.register(
        PromptTemplate(name=PROMPT_NAME, version=PROMPT_VERSION, template=PROMPT_TEMPLATE)
    )


class ReleaseAgentInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    review: ReviewReport
    security_findings: PolicyViolation | None
    docs: DocsPatch
    prior_rejection: str | None = None


class ReleaseAgent:
    """`Agent[ReleaseAgentInput, ReleaseReport]`."""

    name: ClassVar[str] = "release"
    input_model: ClassVar[type[BaseModel]] = ReleaseAgentInput
    output_model: ClassVar[type[BaseModel]] = ReleaseReport
    capabilities: ClassVar[CapabilityManifest] = CapabilityManifest(
        actor="release", allowed_tools=frozenset()
    )
    model_needs: ClassVar[ModelNeeds] = ModelNeeds(reasoning="high", structured_output=True)

    REVIEW_NODE_ID: ClassVar[str] = "review"
    SEC_SCAN_NODE_ID: ClassVar[str] = "sec_scan"
    DOCS_NODE_ID: ClassVar[str] = "docs_gen"
    GATE_NODE_ID: ClassVar[str] = "gate3"

    def build_input(self, ctx: AgentContext) -> ReleaseAgentInput:
        review = ctx.retriever.fetch_latest_from(self.REVIEW_NODE_ID)
        docs = ctx.retriever.fetch_latest_from(self.DOCS_NODE_ID)
        assert isinstance(review, ReviewReport)
        assert isinstance(docs, DocsPatch)
        try:
            security_findings = ctx.retriever.fetch_latest_from(self.SEC_SCAN_NODE_ID)
            assert isinstance(security_findings, PolicyViolation)
        except NoArtifactFromNodeError:
            security_findings = None  # a clean scan legitimately produced nothing
        prior_rejection = ctx.retriever.rejection_reason_of(self.GATE_NODE_ID)
        return ReleaseAgentInput(
            review=review,
            security_findings=security_findings,
            docs=docs,
            prior_rejection=prior_rejection,
        )

    async def run(self, ctx: AgentContext, inp: ReleaseAgentInput) -> AgentResult[ReleaseReport]:
        security_summary = (
            f"{inp.security_findings.severity} - {inp.security_findings.message}"
            if inp.security_findings is not None
            else "clean - no findings"
        )
        prior_rejection_section = (
            f"\nA prior release assessment was rejected at Gate 3 for this reason - address it "
            f"directly:\n<<<REJECTION_REASON>>>\n{inp.prior_rejection}\n<<<END_REJECTION_REASON>>>\n"
            if inp.prior_rejection
            else ""
        )
        artifact, usage = await ctx.complete(
            prompt_name=PROMPT_NAME,
            prompt_version=PROMPT_VERSION,
            variables={
                "review_verdict": inp.review.verdict,
                "review_findings": "; ".join(f.summary for f in inp.review.findings) or "(none)",
                "security_summary": security_summary,
                "docs_summary": inp.docs.summary or "(none)",
                "prior_rejection_section": prior_rejection_section,
            },
            output_schema=ReleaseReport,
            model_needs=self.model_needs,
            # 8192, not 4096 - see agents/architect.py's fuller note on why.
            max_tokens=8192,
        )
        assert isinstance(artifact, ReleaseReport)
        citations = [
            Citation(source=f"artifact:{self.REVIEW_NODE_ID}"),
            Citation(source=f"artifact:{self.DOCS_NODE_ID}"),
        ]
        if inp.security_findings is not None:
            citations.append(Citation(source=f"artifact:{self.SEC_SCAN_NODE_ID}"))
        return AgentResult(
            artifact=artifact,
            confidence=0.6,
            rationale=(
                f"Assessed via {PROMPT_NAME}@v{PROMPT_VERSION}; ready={artifact.ready!r}. "
                "This is a recommendation for Gate 3's human reviewer, not a release decision."
            ),
            citations=tuple(citations),
            usage=usage,
        )


_: Agent[ReleaseAgentInput, ReleaseReport] = ReleaseAgent()
