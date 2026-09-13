"""Adapts one `Agent` into the scheduler's `NodeExecutor` protocol.

Lives here, in `agents/`, and never in `kernel/` - `kernel.scheduler` must
not import the agent plane (`tests/invariants/test_layering.py`). A
`Scheduler` is handed a plain `Mapping[str, NodeExecutor]`; this module is
what makes one entry in that mapping wrap a real `Agent` rather than the
`FixedExecutor`/`SequenceExecutor` fakes Phase 1's tests use.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from ases.agents.base import Agent, AgentContext, AgentExecutionError
from ases.context.retriever import ContextRetriever
from ases.kernel.graph import NodeSpec
from ases.kernel.scheduler import NodeExecutionOutcome
from ases.kernel.state import RunState
from ases.kernel.tools.registry import ToolRegistry
from ases.providers.base import LLMProvider, ProviderError
from ases.providers.prompts.registry import PromptRegistry
from ases.providers.router import ModelRouter
from ases.providers.structured import StructuredOutputExhaustedError


class AgentNodeExecutor:
    """One instance per agent. The scheduler's `executors` mapping has one
    entry per graph `handler` name (docs/03 section 3.3), each wrapping a
    different `Agent` with the same provider/router/prompt registry.

    `tools`/`tool_cwd` are optional - only an agent that actually calls
    `ctx.invoke_tool` (e.g. `scaffold`, via `dotnet.new`) needs them wired;
    `requirements` and `architect` run identically whether or not they are
    supplied, since they never touch `ctx.tools`.
    """

    def __init__(
        self,
        agent: Agent[Any, Any],
        *,
        provider: LLMProvider,
        router: ModelRouter,
        prompts: PromptRegistry,
        run_input: dict[str, str] | None = None,
        tools: ToolRegistry | None = None,
        tool_cwd: PurePosixPath | None = None,
    ) -> None:
        self._agent = agent
        self._provider = provider
        self._router = router
        self._prompts = prompts
        self._run_input = run_input
        self._tools = tools
        self._tool_cwd = tool_cwd

    async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome:
        ctx = AgentContext(
            run_id=str(state.run_id),
            node_id=node.id,
            retriever=ContextRetriever(state),
            provider=self._provider,
            router=self._router,
            prompts=self._prompts,
            run_input=self._run_input,
            tools=self._tools,
            tool_cwd=self._tool_cwd,
            capabilities=self._agent.capabilities,
        )
        try:
            inp = self._agent.build_input(ctx)
            result = await self._agent.run(ctx, inp)
        except (StructuredOutputExhaustedError, ProviderError, AgentExecutionError) as exc:
            # An LLM-boundary failure (bounded repair exhausted, or the
            # provider itself failed), or any `AgentExecutionError` an agent
            # raised for its own normal failure modes (a self-escalation
            # attempt its manifest refused, a tool call that failed) is a
            # node failure, not a crash - the scheduler's own `ON_FAILURE` /
            # repair-cycle machinery (Phase 1) decides what happens next,
            # exactly as it does for a tool failure. Any other exception is a
            # real bug (e.g. a `build_input` wiring mistake) and is left to
            # propagate rather than silently swallowed into "ok=False".
            return NodeExecutionOutcome(ok=False, error=str(exc))

        return NodeExecutionOutcome(
            ok=True,
            artifact_kind=type(result.artifact).__name__,
            artifact_payload=result.artifact.model_dump(mode="json"),
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            usd=result.usage.usd,
        )
