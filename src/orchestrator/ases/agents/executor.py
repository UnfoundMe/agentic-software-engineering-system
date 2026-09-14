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

from ases.agents.base import (
    Agent,
    AgentContext,
    AgentExecutionError,
    CapabilityDeniedError,
)
from ases.context.retriever import ContextRetriever
from ases.kernel.failures import FailureKind
from ases.kernel.graph import NodeSpec
from ases.kernel.scheduler import NodeExecutionOutcome
from ases.kernel.state import RunState
from ases.kernel.tools.registry import ToolRegistry
from ases.providers.base import LLMProvider, ProviderError, TruncatedCompletionError
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
        except (StructuredOutputExhaustedError, TruncatedCompletionError) as exc:
            # The model answered in a way the agent plane cannot turn into an
            # artifact - it failed schema validation, or never finished
            # emitting output at all. Classified apart from every other
            # failure because it says nothing about the code being built: the
            # tools were never reached. Live run `009ea59f-...` reported this
            # as an "Invalid JSON" error and then halted blaming the *build*
            # node's exhausted cycle_budget, which sent diagnosis in entirely
            # the wrong direction. See `kernel.failures`.
            return NodeExecutionOutcome(
                ok=False, failure_kind=FailureKind.AGENT_PROTOCOL_FAILURE, error=str(exc)
            )
        except ProviderError as exc:
            # The provider could not be reached or refused the request: the
            # system failed, not the work.
            return NodeExecutionOutcome(
                ok=False, failure_kind=FailureKind.ORCHESTRATION_FAILURE, error=str(exc)
            )
        except CapabilityDeniedError as exc:
            return NodeExecutionOutcome(
                ok=False, failure_kind=FailureKind.POLICY_DENIED, error=str(exc)
            )
        except AgentExecutionError as exc:
            # Any other failure mode an agent raises for itself - most often a
            # tool call that failed. Not a crash: the scheduler's `ON_FAILURE`
            # / repair-cycle machinery decides what happens next, exactly as
            # it does for a tool node. Any exception *not* listed here is a
            # real bug (e.g. a `build_input` wiring mistake) and is left to
            # propagate rather than silently swallowed into "ok=False".
            return NodeExecutionOutcome(
                ok=False, failure_kind=FailureKind.TOOL_FAILURE, error=str(exc)
            )

        return NodeExecutionOutcome(
            ok=True,
            artifact_kind=type(result.artifact).__name__,
            artifact_payload=result.artifact.model_dump(mode="json"),
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            usd=result.usage.usd,
        )
