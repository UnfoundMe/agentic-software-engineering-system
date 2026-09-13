"""End-to-end: `req -> Gate 1 -> arch -> Gate 2`, two real agents chained
through a live `Scheduler` - the two-gate slice of `workflows/greenfield.yaml`
built so far. Proves `architect` genuinely reads `requirements`'s *content*
(not a fake/hard-coded input) through the mechanism Phase 4 built:
`ContextRetriever.fetch_latest_from`.
"""

from __future__ import annotations

from uuid import uuid4

from ases.agents.architect import ArchitectAgent
from ases.agents.architect import register_prompts as register_architect_prompts
from ases.agents.executor import AgentNodeExecutor
from ases.agents.requirements import RequirementsAgent
from ases.agents.requirements import register_prompts as register_req_prompts
from ases.context.retriever import ArtifactRef, ContextRetriever
from ases.contracts.artifacts import DesignSpec, RequirementSpec
from ases.kernel.gates import ApprovalDecision, BudgetEntryGate, BudgetLimits
from ases.kernel.graph import Edge, NodeKind, NodeSpec, WorkflowGraph
from ases.kernel.scheduler import Scheduler
from ases.kernel.state import NodeStatus, RunStatus
from ases.kernel.store.jsonl import JsonlEventStore
from ases.providers.base import CompletionResult
from ases.providers.mock import MockProvider
from ases.providers.prompts.registry import PromptRegistry
from ases.providers.router import ModelRouter
from tests.unit.fakes import ScriptedApprovals

GENEROUS = BudgetLimits(max_tokens=10_000_000, max_usd=1_000.0, max_wallclock_seconds=3600)


def _graph() -> WorkflowGraph:
    return WorkflowGraph(
        name="req_arch",
        entry=("req",),
        nodes=(
            NodeSpec(id="req", kind=NodeKind.AGENT, handler="req", produces="RequirementSpec"),
            NodeSpec(id="gate1", kind=NodeKind.GATE, requires_approval=True),
            NodeSpec(id="arch", kind=NodeKind.AGENT, handler="arch", produces="DesignSpec"),
            NodeSpec(id="gate2", kind=NodeKind.GATE, requires_approval=True),
            NodeSpec(id="done", kind=NodeKind.TERMINAL),
        ),
        edges=(
            Edge(source="req", target="gate1"),
            Edge(source="gate1", target="arch"),
            Edge(source="arch", target="gate2"),
            Edge(source="gate2", target="done"),
        ),
    )


async def test_architect_reads_the_content_requirements_actually_produced(
    tmp_path: object,
) -> None:
    store = JsonlEventStore(tmp_path)  # type: ignore[arg-type]
    prompts = PromptRegistry()
    register_req_prompts(prompts)
    register_architect_prompts(prompts)

    req_spec = RequirementSpec(
        summary="a url shortener with base62 codes",
        in_scope=("create", "redirect"),
        source_text="Build a URL shortener.",
    )
    design = DesignSpec(
        summary="4-project layered ASP.NET Core solution",
        layers=("Domain", "Application", "Infrastructure", "Api"),
        key_decisions=("EF Core for persistence",),
    )

    req_provider = MockProvider()
    req_provider.respond_with(
        CompletionResult(
            text=req_spec.model_dump_json(),
            parsed=req_spec,
            model_id="m",
            stop_reason="end_turn",
        )
    )
    arch_provider = MockProvider()
    arch_provider.respond_with(
        CompletionResult(
            text=design.model_dump_json(), parsed=design, model_id="m", stop_reason="end_turn"
        )
    )

    req_executor = AgentNodeExecutor(
        RequirementsAgent(),
        provider=req_provider,
        router=ModelRouter(),
        prompts=prompts,
        run_input={"requirement_text": "Build a URL shortener."},
    )
    arch_executor = AgentNodeExecutor(
        ArchitectAgent(), provider=arch_provider, router=ModelRouter(), prompts=prompts
    )

    approvals = ScriptedApprovals(
        {
            "gate1": [ApprovalDecision(granted=True, actor="reviewer")],
            "gate2": [ApprovalDecision(granted=True, actor="reviewer")],
        }
    )
    scheduler = Scheduler(
        _graph(),
        store,
        {"req": req_executor, "arch": arch_executor},
        entry_gate=BudgetEntryGate(GENEROUS),
        approvals=approvals,
    )

    run_id = uuid4()
    state = await scheduler.run(run_id)

    assert state.status is RunStatus.COMPLETED
    assert state.nodes["req"].status is NodeStatus.SUCCEEDED
    assert state.nodes["arch"].status is NodeStatus.SUCCEEDED
    assert state.approvals["gate1"].granted
    assert state.approvals["gate2"].granted

    retriever = ContextRetriever(state)
    design_hash = state.nodes["arch"].produced[0]
    fetched_design = retriever.fetch(ArtifactRef(artifact_hash=design_hash))
    assert fetched_design == design

    # The proof that matters: architect's input was genuinely derived from
    # requirements' real output, via the retriever - not a fixture the test
    # handed it directly. arch_provider only ever saw one canned response
    # (design), so if build_input had failed to find req's content, the run
    # would have raised NoArtifactFromNodeError long before this point.
    req_hash = state.nodes["req"].produced[0]
    assert retriever.fetch(ArtifactRef(artifact_hash=req_hash)) == req_spec


async def test_gate2_rejection_does_not_complete_the_run(tmp_path: object) -> None:
    store = JsonlEventStore(tmp_path)  # type: ignore[arg-type]
    prompts = PromptRegistry()
    register_req_prompts(prompts)
    register_architect_prompts(prompts)

    req_spec = RequirementSpec(summary="s", source_text="raw")
    design = DesignSpec(summary="d")
    req_provider = MockProvider()
    req_provider.respond_with(
        CompletionResult(
            text=req_spec.model_dump_json(), parsed=req_spec, model_id="m", stop_reason="end_turn"
        )
    )
    arch_provider = MockProvider()
    arch_provider.respond_with(
        CompletionResult(
            text=design.model_dump_json(), parsed=design, model_id="m", stop_reason="end_turn"
        )
    )

    req_executor = AgentNodeExecutor(
        RequirementsAgent(),
        provider=req_provider,
        router=ModelRouter(),
        prompts=prompts,
        run_input={"requirement_text": "x"},
    )
    arch_executor = AgentNodeExecutor(
        ArchitectAgent(), provider=arch_provider, router=ModelRouter(), prompts=prompts
    )
    approvals = ScriptedApprovals(
        {
            "gate1": [ApprovalDecision(granted=True, actor="reviewer")],
            "gate2": [ApprovalDecision(granted=False, actor="reviewer", reason="wrong layering")],
        }
    )
    scheduler = Scheduler(
        _graph(),
        store,
        {"req": req_executor, "arch": arch_executor},
        entry_gate=BudgetEntryGate(GENEROUS),
        approvals=approvals,
    )

    state = await scheduler.run(uuid4())

    assert state.status is not RunStatus.COMPLETED
    assert state.approvals["gate2"].granted is False
    assert state.approvals["gate2"].reason == "wrong layering"
