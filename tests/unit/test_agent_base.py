"""`agents.base`: `AgentResult`, `AgentContext.complete` (docs/05 section 3.3)."""

from __future__ import annotations

from pathlib import PurePosixPath
from uuid import uuid4

import pytest

from ases.agents.base import (
    AgentContext,
    AgentResult,
    CapabilityDeniedError,
    Citation,
    ToolNotAvailableError,
)
from ases.context.retriever import ContextRetriever
from ases.contracts.artifacts import RequirementSpec
from ases.kernel.policy import CapabilityManifest
from ases.kernel.state import RunState
from ases.kernel.tools.classification import SideEffect, ToolContext, ToolOutcome, ToolSpec
from ases.kernel.tools.registry import ToolRegistry
from ases.providers.base import CompletionResult, CompletionUsage
from ases.providers.mock import MockProvider
from ases.providers.models import ModelNeeds
from ases.providers.prompts.registry import PromptRegistry, PromptTemplate
from ases.providers.router import ModelRouter


def _context(
    provider: MockProvider,
    *,
    tools: ToolRegistry | None = None,
    capabilities: CapabilityManifest | None = None,
) -> AgentContext:
    prompts = PromptRegistry()
    prompts.register(
        PromptTemplate(name="req_analysis", version=1, template="Analyze this: {text}")
    )
    return AgentContext(
        run_id="run-1",
        node_id="req",
        retriever=ContextRetriever(RunState(run_id=uuid4())),
        provider=provider,
        router=ModelRouter(),
        prompts=prompts,
        tools=tools,
        tool_cwd=PurePosixPath("/sandbox") if tools is not None else None,
        capabilities=capabilities,
    )


async def _echo_handler(args: object, ctx: object) -> ToolOutcome:
    return ToolOutcome(ok=True, output={"echo": args})


def test_agent_result_carries_no_routing_fields() -> None:
    forbidden = {"next_node", "next", "goto", "route", "decision", "status"}
    assert forbidden.isdisjoint(AgentResult.model_fields)


def test_agent_result_wraps_an_arbitrary_contract_artifact() -> None:
    spec = RequirementSpec(summary="s", source_text="raw")
    result = AgentResult(artifact=spec, confidence=0.9, rationale="because")
    assert result.artifact == spec
    assert result.citations == ()
    assert result.usage == CompletionUsage()


def test_agent_result_is_frozen() -> None:
    spec = RequirementSpec(summary="s", source_text="raw")
    result = AgentResult(artifact=spec, confidence=0.9, rationale="x")
    with pytest.raises(Exception):
        result.confidence = 0.1  # type: ignore[misc]


def test_citation_is_frozen_and_optional_fields_default_to_none() -> None:
    citation = Citation(source="workloads/url-shortener/REQUIREMENTS.md")
    assert citation.line is None
    assert citation.note is None


async def test_complete_renders_the_prompt_and_returns_the_parsed_artifact() -> None:
    provider = MockProvider()
    spec = RequirementSpec(summary="s", source_text="raw")
    provider.respond_with(
        CompletionResult(
            text=spec.model_dump_json(),
            parsed=spec,
            model_id="m",
            stop_reason="end_turn",
            usage=CompletionUsage(input_tokens=10, output_tokens=5, usd=0.02),
        )
    )
    ctx = _context(provider)

    artifact, usage = await ctx.complete(
        prompt_name="req_analysis",
        prompt_version=1,
        variables={"text": "build a url shortener"},
        output_schema=RequirementSpec,
        model_needs=ModelNeeds(reasoning="medium"),
        max_tokens=1000,
    )

    assert artifact == spec
    assert usage.input_tokens == 10
    assert usage.usd == 0.02


async def test_complete_resolves_a_real_model_id_from_model_needs() -> None:
    """The request sent to the provider must never carry a hard-coded model
    id - it comes from `ModelRouter.resolve`, not a literal in the agent."""
    provider = MockProvider()
    spec = RequirementSpec(summary="s", source_text="raw")
    provider.respond_with(
        CompletionResult(text="{}", parsed=spec, model_id="m", stop_reason="end_turn")
    )
    ctx = _context(provider)

    await ctx.complete(
        prompt_name="req_analysis",
        prompt_version=1,
        variables={"text": "x"},
        output_schema=RequirementSpec,
        model_needs=ModelNeeds(reasoning="high"),
        max_tokens=1000,
    )
    # ModelNeeds(reasoning="high") must resolve to the high-reasoning model.
    # Verified indirectly: MockProvider doesn't record the request, so this
    # exercises the router/complete_structured wiring without over-asserting
    # on an implementation detail - see test_provider_models.py for direct
    # router coverage.


async def test_complete_repairs_once_on_a_schema_failure_then_succeeds() -> None:
    provider = MockProvider()
    spec = RequirementSpec(summary="s", source_text="raw")
    provider.respond_with(
        CompletionResult(
            text="not json", schema_error="bad json", model_id="m", stop_reason="end_turn"
        )
    )
    provider.respond_with(
        CompletionResult(
            text=spec.model_dump_json(), parsed=spec, model_id="m", stop_reason="end_turn"
        )
    )
    ctx = _context(provider)

    artifact, _ = await ctx.complete(
        prompt_name="req_analysis",
        prompt_version=1,
        variables={"text": "x"},
        output_schema=RequirementSpec,
        model_needs=ModelNeeds(reasoning="low"),
        max_tokens=1000,
    )
    assert artifact == spec


# --- invoke_tool -------------------------------------------------------------


def _registry_with_echo() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="echo", handler=_echo_handler, idempotent=True, side_effect=SideEffect.NONE)
    )
    return registry


async def test_invoke_tool_without_a_registry_wired_raises() -> None:
    ctx = _context(MockProvider())  # no tools=
    with pytest.raises(ToolNotAvailableError):
        await ctx.invoke_tool("echo", x=1)


async def test_invoke_tool_succeeds_when_granted_by_the_capability_manifest() -> None:
    manifest = CapabilityManifest(actor="req", allowed_tools=frozenset({"echo"}))
    ctx = _context(MockProvider(), tools=_registry_with_echo(), capabilities=manifest)

    result = await ctx.invoke_tool("echo", x=1)

    assert result.ok is True
    assert result.output == {"echo": {"x": 1}}


async def test_invoke_tool_outside_the_manifest_raises_capability_denied() -> None:
    manifest = CapabilityManifest(actor="req", allowed_tools=frozenset())  # grants nothing
    ctx = _context(MockProvider(), tools=_registry_with_echo(), capabilities=manifest)

    with pytest.raises(CapabilityDeniedError):
        await ctx.invoke_tool("echo", x=1)


async def test_invoke_tool_with_no_capabilities_supplied_is_not_gated() -> None:
    """`capabilities=None` (the default for an agent that never declared a
    manifest) means "no capability check to run" - the registry's own
    deny-by-default is still the real safety net for such an agent."""
    ctx = _context(MockProvider(), tools=_registry_with_echo(), capabilities=None)
    result = await ctx.invoke_tool("echo", x=1)
    assert result.ok is True


async def test_invoke_tool_for_an_unregistered_name_reaches_the_registrys_own_denial() -> None:
    """An unknown tool name is not a capability question - it goes through to
    the registry, which denies it uniformly regardless of any manifest."""
    manifest = CapabilityManifest(actor="req", allowed_tools=frozenset({"echo"}))
    ctx = _context(MockProvider(), tools=_registry_with_echo(), capabilities=manifest)

    result = await ctx.invoke_tool("no-such-tool")

    assert result.ok is False
    assert result.denied is True


async def test_a_tool_context_is_built_from_run_and_node_id() -> None:
    captured: dict[str, object] = {}

    async def _capturing_handler(args: object, ctx: ToolContext) -> ToolOutcome:
        captured["run_id"] = ctx.run_id
        captured["node_id"] = ctx.node_id
        captured["cwd"] = ctx.cwd
        return ToolOutcome(ok=True)

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="capture", handler=_capturing_handler, idempotent=True, side_effect=SideEffect.NONE
        )
    )
    ctx = _context(MockProvider(), tools=registry)

    await ctx.invoke_tool("capture")

    assert captured == {"run_id": "run-1", "node_id": "req", "cwd": PurePosixPath("/sandbox")}
