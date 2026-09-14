"""The LLM boundary's shared vocabulary: requests, results, and the provider
protocol every adapter (Anthropic, cassette, mock) implements identically.

Agents never talk to a vendor SDK directly - they go through `LLMProvider`,
which is why swapping Anthropic for a cassette or a mock is a constructor
change, never a call-site change. This module intentionally knows nothing
about Anthropic, Pydantic schemas for a *specific* artifact, or how a model
was chosen (that is `router.py`) - it is the seam, not any one side of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

Role = Literal["user", "assistant"]


class Message(BaseModel):
    """One turn of a conversation. Deliberately just role + text: tool use and
    multi-block content belong to a later phase that actually needs them
    (Phase 4's agents), not to the boundary itself."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str


class CompletionRequest(BaseModel):
    """Everything a provider needs to produce one completion.

    `model_id` is carried here (resolved by `ModelRouter` at the call site,
    never chosen by the provider) but is deliberately **excluded** from the
    cassette key - see `cassette.cassette_key` - so a model swap in the router
    does not invalidate recorded cassettes.

    `output_schema`, when set, asks the provider to constrain and validate its
    response against that Pydantic model (docs/05 section 3.4: structured
    output validation). `prompt_version` identifies the prompt template that
    produced `rendered_prompt` - see `prompts/registry.py`.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model_id: str
    prompt_version: str
    rendered_prompt: str
    system: str | None = None
    history: tuple[Message, ...] = ()
    output_schema: type[BaseModel] | None = None
    max_tokens: int

    @property
    def messages(self) -> tuple[Message, ...]:
        return (*self.history, Message(role="user", content=self.rendered_prompt))


class CompletionUsage(BaseModel):
    """Token and cost metadata for one completion.

    Shaped to map directly onto an `LLM_COMPLETED` event payload
    (`kernel.events.EventType.LLM_COMPLETED`) once an executor emits one -
    field names match `kernel.state.Usage` deliberately.
    """

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0


class CompletionResult(BaseModel):
    """What every provider returns, regardless of vendor.

    `parsed` is `None` whenever `output_schema` was not requested, and also
    when it *was* requested but the raw output failed to validate against it -
    the two are distinguished by `schema_error`, which `structured.py` uses to
    decide whether a bounded repair attempt is warranted. A provider must
    never raise on a schema-validation failure; that failure is data the
    caller acts on, not an exception the caller must catch.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    text: str
    parsed: BaseModel | None = None
    schema_error: str | None = None
    #: The response ran out of `max_tokens` before the model finished. A
    #: distinct condition from `schema_error`, and one a repair attempt must
    #: handle differently: re-sending the same prompt under the same ceiling
    #: truncates again at the same place, which is exactly what live run
    #: `009ea59f-...` did - twice, producing two byte-identical "Invalid
    #: JSON" errors that were neither about JSON nor about the prompt. See
    #: `providers.structured.complete_structured`.
    truncated: bool = False
    model_id: str
    stop_reason: str
    usage: CompletionUsage = CompletionUsage()
    raw: Mapping[str, Any] = {}


class TruncatedCompletionError(RuntimeError):
    """The model ran out of `max_tokens` before producing usable output, and
    raising the ceiling once did not change that.

    Distinct from `structured.StructuredOutputExhaustedError` (the model
    answered, but wrongly) and from `ProviderError` (no answer at all,
    transport-level). Carries its own type so `agents.executor` can classify
    it as `FailureKind.AGENT_PROTOCOL_FAILURE` with a message that names the
    real cause instead of a misleading JSON-parse error."""

    def __init__(self, schema_name: str, attempts: str) -> None:
        super().__init__(
            f"{schema_name}: the model's response was truncated at max_tokens before "
            f"usable output was produced, and raising the ceiling did not help ({attempts}). "
            "This is an agent protocol failure, not a source-code failure."
        )
        self.schema_name = schema_name


class ProviderError(RuntimeError):
    """A provider could not produce a completion at all (network, auth, rate
    limit). Distinct from a schema-validation failure, which is not an error -
    see `CompletionResult.schema_error`."""


@runtime_checkable
class LLMProvider(Protocol):
    """The one interface every adapter implements. Agents (Phase 4) depend on
    this and never on a concrete adapter, so `ASES_LLM_MODE` can swap
    Anthropic for cassette replay or a mock with no call-site change."""

    async def complete(self, request: CompletionRequest) -> CompletionResult: ...
