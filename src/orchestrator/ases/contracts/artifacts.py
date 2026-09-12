"""The artifact contracts, one per SDLC stage.

Grouped by lifecycle stage to match docs/03 section 3.3's node choreography
and `workflows/greenfield.yaml`'s `produces:` fields - each model here is one
of those. The `CONTRACTS` registry at the bottom is what
`tests/unit/test_contracts.py` uses to prove every kind a workflow YAML names
actually has a model, so the two cannot silently drift apart.

Field sets are deliberately modest. These are the first draft of a stable
interface for agents that do not exist yet (Phase 4); getting every field
right before a single agent has been built would be guessing. What matters
now is that every kind named in the workflow graph has *a* contract, and that
L1 schema validation (docs/05 section 6) has something real to check against.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from ases.contracts.base import ArtifactModel

# --- requirements -----------------------------------------------------


class Ambiguity(ArtifactModel):
    question: str
    resolution: str | None = None
    rationale: str | None = None
    resolved_by: str | None = None
    resolved_at: datetime | None = None

    @property
    def is_resolved(self) -> bool:
        return self.resolution is not None


class RequirementSpec(ArtifactModel):
    """Output of the requirements agent: the normalized engineering problem."""

    summary: str
    in_scope: tuple[str, ...] = ()
    out_of_scope: tuple[str, ...] = ()
    source_text: str


class AmbiguityRegister(ArtifactModel):
    """Every ambiguity the requirements agent surfaced, resolved or not.

    A human gate (docs/02 section 4.4) reviews this alongside `RequirementSpec`
    before the run may proceed - an unresolved ambiguity here is what Gate 1
    is checking for.
    """

    ambiguities: tuple[Ambiguity, ...] = ()

    @property
    def all_resolved(self) -> bool:
        return all(a.is_resolved for a in self.ambiguities)


# --- architecture (brownfield impact is Stage 2; the model exists now) -


class ImpactReport(ArtifactModel):
    """Stage 2 (Roslyn) output. Modelled now so the contract is stable before
    the analyzer exists - see docs/02 Phase 6."""

    impacted_modules: tuple[str, ...] = ()
    impacted_endpoints: tuple[str, ...] = ()
    impacted_tables: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class ArchitectureDecision(ArtifactModel):
    """One ADR. `DesignSpec` and its ADRs are produced by the same node
    (docs/03 section 3.3: 'DesignSpec + ADR-001') but are distinct artifacts,
    hashed and versioned independently."""

    title: str
    status: str = "proposed"
    context: str
    decision: str
    consequences: str = ""


class DesignSpec(ArtifactModel):
    """Output of the architect agent. Must state how the chosen structure
    satisfies claude.md section 9's layer separation (docs/03 section 3.1a) -
    that is guidance given to the agent, not a field this model enforces."""

    summary: str
    layers: tuple[str, ...] = ()
    key_decisions: tuple[str, ...] = ()


class SolutionSkeleton(ArtifactModel):
    """Output of `SCAFFOLD` (docs/02 section 7.1): the frozen project
    structure and interface signatures that parallel implementation tasks
    build against - this is what removes interface drift as a failure mode."""

    projects: tuple[str, ...] = ()
    pinned_packages: tuple[str, ...] = ()
    frozen_interfaces: tuple[str, ...] = ()


# --- planning -----------------------------------------------------------


class TaskSpec(ArtifactModel):
    id: str
    description: str
    depends_on: tuple[str, ...] = ()


class TaskGraph(ArtifactModel):
    """Output of the decomposer. In the running scheduler this becomes a
    subgraph proposal (`WorkflowGraph.with_subgraph`) - not yet wired to a
    live run; see `kernel/scheduler.py`'s module docstring."""

    tasks: tuple[TaskSpec, ...] = ()


# --- implementation -------------------------------------------------------


class FileChange(ArtifactModel):
    path: str
    change_kind: str = "modify"  # add | modify | delete
    content: str | None = None
    diff: str | None = None


class CodePatch(ArtifactModel):
    """Output of an implementer or repair agent. Applied only inside the
    sandbox (docs/05 section 3.5) - never to the real repository."""

    summary: str
    files: tuple[FileChange, ...] = ()


class DocsPatch(ArtifactModel):
    """Output of the docs agent: documentation changes, structured the same
    way as a `CodePatch` since both land through the same sandbox promotion
    path."""

    summary: str
    files: tuple[FileChange, ...] = ()


# --- testing and review ----------------------------------------------------


class TestSuite(ArtifactModel):
    summary: str
    files: tuple[FileChange, ...] = ()
    coverage_delta: float | None = None


class Finding(ArtifactModel):
    """One reviewer observation. `citation` is mandatory in spirit (docs/05
    section 3.3 requires the critic to cite file:line) though not enforced by
    this schema alone - L4 semantic validation checks that separately."""

    summary: str
    citation: str | None = None
    severity: str = "info"  # info | warning | high


class ReviewReport(ArtifactModel):
    findings: tuple[Finding, ...] = ()
    verdict: str = "pass"  # pass | concerns | block


class PolicyViolation(ArtifactModel):
    """A policy engine (Phase 3) finding, or - for now - a static security
    scan result. Distinct from the kernel's `POLICY_VIOLATION` event: this is
    the *content* an agent/tool produced; the event is the kernel's record
    that it happened."""

    rule: str
    severity: str = "medium"  # low | medium | high
    message: str
    node_id: str | None = None


# --- release --------------------------------------------------------------


class ReleaseReport(ArtifactModel):
    ready: bool
    checklist: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()


class RunSummary(ArtifactModel):
    """The final engineering summary (the brief's Feature 8), generated from
    the event log rather than hand-written - docs/05 section 6."""

    plan_and_rationale: str
    artifacts: tuple[str, ...] = ()  # artifact hashes referenced by the run
    risks: tuple[str, ...] = ()
    validation: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


# --- registry ---------------------------------------------------------

#: kind-string (as recorded in ARTIFACT_PRODUCED events and NodeSpec.produces)
#: -> the model it must validate against. Kept in sync with every workflow
#: YAML's `produces:` field by `tests/unit/test_contracts.py`.
CONTRACTS: dict[str, type[ArtifactModel]] = {
    "RequirementSpec": RequirementSpec,
    "AmbiguityRegister": AmbiguityRegister,
    "ImpactReport": ImpactReport,
    "ArchitectureDecision": ArchitectureDecision,
    "DesignSpec": DesignSpec,
    "SolutionSkeleton": SolutionSkeleton,
    "TaskGraph": TaskGraph,
    "CodePatch": CodePatch,
    "DocsPatch": DocsPatch,
    "TestSuite": TestSuite,
    "ReviewReport": ReviewReport,
    "PolicyViolation": PolicyViolation,
    "ReleaseReport": ReleaseReport,
    "RunSummary": RunSummary,
}
