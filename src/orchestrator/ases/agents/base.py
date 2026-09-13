"""The agent plane's own vocabulary (docs/05 section 3.3).

`AgentResult` is deliberately narrow: an artifact, a confidence score, a
rationale, citations, and usage - nothing that could let an agent steer the
graph. `tests/invariants/test_layering.py::test_agent_result_cannot_carry_a_routing_decision`
asserts this by inspecting `AgentResult.model_fields` directly, so the
absence of a `next_node`/`status`/`decision`-shaped field is enforced, not
just documented.

`AgentContext` is what an agent actually touches: a scoped view of prior
artifacts (`ContextRetriever`), one convenience method to call the LLM
through the Phase 2 boundary (`complete_structured`) with the right model
already resolved, and - for an agent whose `capabilities` grant it any -
`invoke_tool` to reach the Phase 3 tool registry. An agent never imports
`providers.anthropic_provider` or `kernel.tools.registry` directly - only
this context and the protocols/functions it wraps.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import ClassVar, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ases.context.retriever import ContextRetriever
from ases.kernel.events import ActorKind
from ases.kernel.policy import Action, CapabilityManifest, PolicyDecision, enforce_capability
from ases.kernel.tools.classification import ToolContext
from ases.kernel.tools.registry import ToolInvocationResult, ToolRegistry, UnknownToolNameError
from ases.providers.base import CompletionRequest, CompletionUsage, LLMProvider
from ases.providers.models import ModelNeeds
from ases.providers.prompts.registry import PromptRegistry
from ases.providers.router import ModelRouter
from ases.providers.structured import complete_structured

#: Both invariant: `TIn` appears in both a return position (`build_input`)
#: and a parameter position (`run`) of the same `Agent` protocol, which rules
#: out contravariance; `TOut` is used the same way `AgentResult` (a concrete,
#: non-Protocol generic model) already requires it - see its own note.
TIn = TypeVar("TIn", bound=BaseModel)
TOut = TypeVar("TOut", bound=BaseModel)


class Citation(BaseModel):
    """Evidence for a claim in `AgentResult.rationale` - `file:line` when the
    claim is about generated code, or a source-text reference for anything
    else. Required in spirit of a reviewer/critic agent (docs/05 section 3.3);
    not enforced by this schema alone, since not every agent has code to cite."""

    model_config = ConfigDict(frozen=True)

    source: str
    line: int | None = None
    note: str | None = None


class AgentResult[T: BaseModel](BaseModel):
    """What every agent returns. Nothing here is a routing decision - see
    the module docstring and the invariant test that checks it."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    artifact: T
    confidence: float
    rationale: str
    citations: tuple[Citation, ...] = ()
    usage: CompletionUsage = CompletionUsage()
    # NOTE: there is deliberately no `next_node`, `status` or `decision`
    # field. Adding one would let an agent steer the graph; the invariant
    # test in tests/invariants/test_layering.py guards this directly.


class AgentContext:
    """Scoped services one agent invocation may use. Constructed fresh per
    call by the executor that adapts an `Agent` to the scheduler's
    `NodeExecutor` protocol (`agents/executor.py`) - never held across calls,
    since it is scoped to one run and one node."""

    def __init__(
        self,
        *,
        run_id: str,
        node_id: str,
        retriever: ContextRetriever,
        provider: LLMProvider,
        router: ModelRouter,
        prompts: PromptRegistry,
        run_input: dict[str, str] | None = None,
        tools: ToolRegistry | None = None,
        tool_cwd: PurePosixPath | None = None,
        capabilities: CapabilityManifest | None = None,
    ) -> None:
        self.run_id = run_id
        self.node_id = node_id
        self.retriever = retriever
        self.provider = provider
        self.router = router
        self.prompts = prompts
        #: The run's original launch parameters (e.g. the requirement text
        #: given to `ases run greenfield --requirement ...`). Only ever
        #: consulted by an entry-point agent's `build_input` - every other
        #: agent's input comes from `retriever.fetch`, since it always has a
        #: real upstream artifact to read.
        self.run_input = run_input or {}
        #: Present only for an agent whose graph position actually needs a
        #: tool (e.g. `scaffold`, via `dotnet.new`) - `requirements` and
        #: `architect` are constructed without these and never call
        #: `invoke_tool`. `capabilities` is the agent's *own*
        #: `CapabilityManifest` (docs/05 section 3.3), checked in
        #: `invoke_tool` before the registry is ever reached - CLAUDE.md
        #: section 3's "agents cannot modify their own permissions" made
        #: concrete: this manifest is supplied by the executor, external to
        #: the agent, never something the agent's own code can widen.
        self.tools = tools
        self.tool_cwd = tool_cwd
        self.capabilities = capabilities

    async def complete(
        self,
        *,
        prompt_name: str,
        prompt_version: int | None,
        variables: dict[str, str],
        output_schema: type[BaseModel],
        model_needs: ModelNeeds,
        max_tokens: int,
        system: str | None = None,
    ) -> tuple[BaseModel, CompletionUsage]:
        """Renders the named prompt, resolves a model from `model_needs`
        (never a hard-coded model id), and returns the schema-validated
        artifact plus the usage incurred - including the repair attempt's
        usage, if one happened, so callers never under-report cost.

        Raises `providers.structured.StructuredOutputExhaustedError` if the
        bounded repair also fails - the caller (an `Agent.run` implementation)
        lets that propagate; the executor adapter turns it into a failed
        `NodeExecutionOutcome`, exactly like any other agent failure.
        """
        template = self.prompts.get(prompt_name, version=prompt_version)
        rendered = template.render(**variables)
        model = self.router.resolve(model_needs)
        request = CompletionRequest(
            model_id=model.id,
            prompt_version=template.prompt_version,
            rendered_prompt=rendered,
            system=system,
            output_schema=output_schema,
            max_tokens=max_tokens,
        )
        result = await complete_structured(self.provider, request)
        assert result.parsed is not None  # complete_structured guarantees this on success
        return result.parsed, result.usage

    async def invoke_tool(self, tool_name: str, **args: object) -> ToolInvocationResult:
        """Reaches the Phase 3 tool registry, gated by this agent's own
        `CapabilityManifest` first - a self-escalation attempt (a tool name
        outside what the executor granted this agent) is refused here,
        before the registry's own deny-by-default check ever runs, and
        raises rather than returning a normal denial result: unlike an
        unknown tool name (a registry-level fact, uniform for every caller),
        this is specifically *this agent* asking for *more than it was
        granted* - an authoring bug worth failing loudly on, not routing
        through the same result type an ordinary tool failure uses.

        The parameter is named `tool_name`, not `name`, deliberately: several
        real tools (`dotnet.new`) take a `name` argument of their own in
        `**args`, and a caller must be able to pass `name=...` through
        without it colliding with this method's own tool-selector parameter.
        """
        if self.tools is None or self.tool_cwd is None:
            raise ToolNotAvailableError(
                f"agent {self.node_id!r} attempted to invoke tool {tool_name!r}, but this "
                "context has no ToolRegistry/sandbox wired in"
            )
        if self.capabilities is not None:
            try:
                spec = self.tools.spec_for(tool_name)
            except UnknownToolNameError:
                spec = None
            if spec is not None:
                action = Action(
                    tool_name=tool_name,
                    side_effect=spec.side_effect,
                    actor_kind=ActorKind.AGENT,
                    requires_approval=spec.requires_approval,
                )
                evaluation = enforce_capability(self.capabilities, action)
                if evaluation.decision is PolicyDecision.DENY:
                    raise CapabilityDeniedError(evaluation.reason)
        tool_ctx = ToolContext(cwd=self.tool_cwd, run_id=self.run_id, node_id=self.node_id)
        return await self.tools.invoke(tool_name, args, tool_ctx)


class AgentExecutionError(RuntimeError):
    """Base for a failure an agent's own `run()` raises that represents a
    normal node failure, not a crash - `agents.executor.AgentNodeExecutor`
    catches this uniformly (alongside the LLM-boundary-specific
    `StructuredOutputExhaustedError`/`ProviderError`) and turns it into a
    failed `NodeExecutionOutcome`, exactly as it does for a tool failure. A
    concrete agent module (e.g. `agents/scaffold.py`) subclasses this for its
    own tool-invocation failures rather than growing the executor's except
    clause by one member per agent."""


class ToolNotAvailableError(RuntimeError):
    """`invoke_tool` was called on a context with no `ToolRegistry`/sandbox
    wired in. Deliberately **not** an `AgentExecutionError`: this is a
    wiring bug (the executor/CLI constructed the agent without the tools it
    declared it needs), not a transient failure a retry could fix, so it is
    left to propagate loudly - the same posture `build_input`'s own errors
    already take (see `test_a_build_input_error_is_not_swallowed`)."""


class CapabilityDeniedError(AgentExecutionError):
    """This agent's own `CapabilityManifest` does not grant the requested
    tool - a self-escalation attempt (CLAUDE.md section 3), refused before
    the tool registry is ever reached."""


@runtime_checkable
class Agent(Protocol[TIn, TOut]):
    """One stateless LLM-backed worker. Produces one artifact; never decides
    what runs next (docs/05 section 3.3).

    `build_input` exists because a `NodeExecutor` (`agents/executor.py`)
    receives only `NodeSpec` and `RunState` - the workflow graph's edges are
    not visible at that point, so nothing generic can know which upstream
    node(s) feed a given agent. Each agent's `build_input` hardcodes that
    knowledge (e.g. the architect agent reads node id `"req"`), which is
    correct rather than a workaround: a workflow's node choreography (docs/03
    section 3.3) is a fixed contract between adjacent agents, known when each
    agent is written, not something to rediscover generically at runtime.
    """

    name: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]
    output_model: ClassVar[type[BaseModel]]
    capabilities: ClassVar[CapabilityManifest]
    model_needs: ClassVar[ModelNeeds]

    def build_input(self, ctx: AgentContext) -> TIn: ...

    async def run(self, ctx: AgentContext, inp: TIn) -> AgentResult[TOut]: ...
