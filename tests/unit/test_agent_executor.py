"""`AgentNodeExecutor`: bridges an `Agent` into a live `Scheduler` run.

The last two tests are the payoff for this session's kernel change (Phase 4
prerequisite: artifact content now flows through the event log) - a real,
end-to-end run: `Scheduler` -> `AgentNodeExecutor` -> `RequirementsAgent` ->
`MockProvider`, then a human gate, then reading the produced `RequirementSpec`
*content* back out via `ContextRetriever.fetch` - not just its hash.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from ases.agents.base import AgentContext, AgentResult
from ases.agents.executor import AgentNodeExecutor
from ases.agents.requirements import RequirementsAgent, register_prompts
from ases.context.retriever import ArtifactRef, ContextRetriever
from ases.contracts.artifacts import RequirementSpec
from ases.kernel.gates import ApprovalDecision, BudgetEntryGate, BudgetLimits
from ases.kernel.graph import Edge, NodeKind, NodeSpec, WorkflowGraph
from ases.kernel.scheduler import Scheduler
from ases.kernel.state import NodeStatus, RunState, RunStatus
from ases.kernel.store.jsonl import JsonlEventStore
from ases.providers.base import CompletionResult, ProviderError
from ases.providers.mock import MockProvider
from ases.providers.prompts.registry import PromptRegistry
from ases.providers.router import ModelRouter
from ases.providers.structured import StructuredOutputExhaustedError
from tests.unit.fakes import ScriptedApprovals

GENEROUS = BudgetLimits(max_tokens=10_000_000, max_usd=1_000.0, max_wallclock_seconds=3600)


def _node(node_id: str = "req") -> NodeSpec:
    return NodeSpec(id=node_id, kind=NodeKind.AGENT, handler=node_id, produces="RequirementSpec")


def _prompts() -> PromptRegistry:
    registry = PromptRegistry()
    register_prompts(registry)
    return registry


async def test_execute_converts_a_successful_agent_result_into_an_ok_outcome() -> None:
    spec = RequirementSpec(summary="s", in_scope=("a",), source_text="raw")
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=spec.model_dump_json(), parsed=spec, model_id="m", stop_reason="end_turn"
        )
    )
    executor = AgentNodeExecutor(
        RequirementsAgent(),
        provider=provider,
        router=ModelRouter(),
        prompts=_prompts(),
        run_input={"requirement_text": "build a thing"},
    )

    outcome = await executor.execute(_node(), RunState(run_id=uuid4()))

    assert outcome.ok is True
    assert outcome.artifact_kind == "RequirementSpec"
    assert outcome.artifact_payload == spec.model_dump(mode="json")


async def test_execute_reports_a_structured_output_exhaustion_as_a_failed_outcome() -> None:
    class _AlwaysFailsAgent:
        name = "always_fails"
        input_model = RequirementsAgent.input_model
        output_model = RequirementSpec
        capabilities = RequirementsAgent.capabilities
        model_needs = RequirementsAgent.model_needs

        def build_input(self, ctx: AgentContext) -> object:
            return object()

        async def run(self, ctx: AgentContext, inp: object) -> AgentResult[RequirementSpec]:
            raise StructuredOutputExhaustedError("RequirementSpec", "bad", "still bad")

    executor = AgentNodeExecutor(
        _AlwaysFailsAgent(),  # type: ignore[arg-type]
        provider=MockProvider(),
        router=ModelRouter(),
        prompts=_prompts(),
    )
    outcome = await executor.execute(_node(), RunState(run_id=uuid4()))

    assert outcome.ok is False
    assert outcome.error is not None
    assert "repair attempt" in outcome.error


async def test_execute_reports_a_provider_error_as_a_failed_outcome() -> None:
    class _RaisesProviderError:
        name = "raises"
        input_model = RequirementsAgent.input_model
        output_model = RequirementSpec
        capabilities = RequirementsAgent.capabilities
        model_needs = RequirementsAgent.model_needs

        def build_input(self, ctx: AgentContext) -> object:
            return object()

        async def run(self, ctx: AgentContext, inp: object) -> AgentResult[RequirementSpec]:
            raise ProviderError("network exploded")

    executor = AgentNodeExecutor(
        _RaisesProviderError(),  # type: ignore[arg-type]
        provider=MockProvider(),
        router=ModelRouter(),
        prompts=_prompts(),
    )
    outcome = await executor.execute(_node(), RunState(run_id=uuid4()))

    assert outcome.ok is False
    assert outcome.error == "network exploded"


async def test_a_build_input_error_is_not_swallowed() -> None:
    """A genuine authoring bug (missing run_input) must surface loudly, not
    be reported as a normal node failure."""
    executor = AgentNodeExecutor(
        RequirementsAgent(), provider=MockProvider(), router=ModelRouter(), prompts=_prompts()
    )
    with pytest.raises(ValueError, match="requirement_text"):
        await executor.execute(_node(), RunState(run_id=uuid4()))


# --- end-to-end: a live Scheduler run through a real agent ------------------


async def test_end_to_end_run_produces_a_requirement_spec_and_its_content_is_retrievable(
    tmp_path: object,
) -> None:
    store = JsonlEventStore(tmp_path)  # type: ignore[arg-type]
    graph = WorkflowGraph(
        name="req_only",
        entry=("req",),
        nodes=(
            _node("req"),
            NodeSpec(id="gate1", kind=NodeKind.GATE, requires_approval=True),
            NodeSpec(id="done", kind=NodeKind.TERMINAL),
        ),
        edges=(Edge(source="req", target="gate1"), Edge(source="gate1", target="done")),
    )

    spec = RequirementSpec(
        summary="a url shortener", in_scope=("create", "redirect"), source_text="raw"
    )
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=spec.model_dump_json(), parsed=spec, model_id="m", stop_reason="end_turn"
        )
    )
    executor = AgentNodeExecutor(
        RequirementsAgent(),
        provider=provider,
        router=ModelRouter(),
        prompts=_prompts(),
        run_input={"requirement_text": "Build a URL shortener."},
    )
    approvals = ScriptedApprovals({"gate1": [ApprovalDecision(granted=True, actor="reviewer")]})
    scheduler = Scheduler(
        graph,
        store,
        {"req": executor},
        entry_gate=BudgetEntryGate(GENEROUS),
        approvals=approvals,
    )

    run_id = uuid4()
    state = await scheduler.run(run_id)

    assert state.status is RunStatus.COMPLETED
    assert state.nodes["req"].status is NodeStatus.SUCCEEDED

    artifact_hash = state.nodes["req"].produced[0]
    retriever = ContextRetriever(state)
    fetched = retriever.fetch(ArtifactRef(artifact_hash=artifact_hash))
    assert fetched == spec


async def test_end_to_end_run_survives_process_restart_via_replay(tmp_path: object) -> None:
    """Phase 1's own resumability property, re-proven with a real agent in
    the loop instead of a fake one: folding the exported event log offline
    reproduces the same artifact content, with no scheduler and no LLM call."""
    store = JsonlEventStore(tmp_path)  # type: ignore[arg-type]
    graph = WorkflowGraph(
        name="req_only",
        entry=("req",),
        nodes=(_node("req"), NodeSpec(id="done", kind=NodeKind.TERMINAL)),
        edges=(Edge(source="req", target="done"),),
    )
    spec = RequirementSpec(summary="s", source_text="raw")
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=spec.model_dump_json(), parsed=spec, model_id="m", stop_reason="end_turn"
        )
    )
    executor = AgentNodeExecutor(
        RequirementsAgent(),
        provider=provider,
        router=ModelRouter(),
        prompts=_prompts(),
        run_input={"requirement_text": "x"},
    )
    run_id = uuid4()
    await Scheduler(graph, store, {"req": executor}, entry_gate=BudgetEntryGate(GENEROUS)).run(
        run_id
    )

    from ases.kernel.state import fold

    events = [e async for e in store.read(run_id)]
    replayed = fold(run_id, events)

    artifact_hash = replayed.nodes["req"].produced[0]
    assert replayed.artifact_content[artifact_hash] == spec.model_dump(mode="json")
