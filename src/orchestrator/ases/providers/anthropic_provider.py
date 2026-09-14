"""The live `LLMProvider` adapter, backed by the Anthropic Messages API.

Used directly under `ASES_LLM_MODE=live`, and wrapped by
`cassette.RecordingProvider` under `ASES_LLM_MODE=record`. Never constructed
under `replay` or `mock` - see `factory.get_provider`.

Structured output deliberately does **not** use the SDK's `messages.parse()`
helper. That helper raises the raw `pydantic.ValidationError` out of the SDK
call itself when the model's JSON fails to validate, which would discard the
response's usage/cost and stop_reason along with it - and usage is billed
regardless of whether the output validated. Instead this adapter builds the
JSON-schema `output_config` itself (docs: "Raw Schema" structured-output
pattern), always gets back a normal `Message` with real usage, and validates
the text against the Pydantic model itself - so a schema failure becomes
`CompletionResult.schema_error` (data `structured.py` can act on) with usage
still attached, never a swallowed exception.

`_create_with_retry` retries a bounded number of times on a transient
connection failure (including a bare `ConnectionError` - see docs/07's live
run notes for the Windows asyncio/SSL defect that raises one) before giving
up as a `ProviderError`. A real API error response (`APIStatusError`) is
never retried here - it fails immediately, unchanged from before.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import anthropic
from anthropic.types import MessageParam
from anthropic.types.output_config_param import OutputConfigParam
from pydantic import ValidationError

from ases.providers.base import (
    CompletionRequest,
    CompletionResult,
    CompletionUsage,
    Message,
    ProviderError,
)
from ases.providers.models import ModelCatalogEntryNotFoundError, spec_for

# A bare `ConnectionError` - not `anthropic.APIConnectionError` - is a known
# Windows asyncio/SSL transport defect (ProactorEventLoop double-invoking
# `connection_lost()`, cpython gh-83413): it surfaces below the layer the
# Anthropic SDK's own retries recognize, on an otherwise-healthy connection.
# Both are transient and worth a bounded retry; see docs/07 for the live run
# this was found on.
_TRANSIENT_ERRORS = (anthropic.APIConnectionError, ConnectionError)
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 1.0


def _to_message_param(message: Message) -> MessageParam:
    return {"role": message.role, "content": message.content}


def _first_text(content: Sequence[Any]) -> str:
    """The first `text` block, or `""` when the response has none.

    An empty return is not a defensive nicety - it is a condition live run
    `009ea59f-...` hit twice. Current-generation models run adaptive extended
    thinking whenever `thinking` is omitted (see the module docstring), and
    thinking tokens are billed against `max_tokens` like any other output. A
    response that exhausts the ceiling while still inside its thinking block
    comes back with content but no `text` block at all. The caller
    distinguishes that from genuinely malformed output via `stop_reason`, not
    by guessing from the empty string.
    """
    for block in content:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
            return str(text)
    return ""


def _usage(usage: Any, model_id: str) -> CompletionUsage:
    input_tokens = int(usage.input_tokens)
    output_tokens = int(usage.output_tokens)
    try:
        spec = spec_for(model_id)
        usd = spec.cost_usd(input_tokens=input_tokens, output_tokens=output_tokens)
    except ModelCatalogEntryNotFoundError:
        # A model id outside the Phase 2 catalog (e.g. called directly in a
        # test, or a future model) still returns real usage - just without a
        # cost figure we have no price for. Silence here would hide spend;
        # zero is the honest "unknown," not a guess.
        usd = 0.0
    return CompletionUsage(input_tokens=input_tokens, output_tokens=output_tokens, usd=usd)


class AnthropicProvider:
    """Live `LLMProvider`. See module docstring for the structured-output design."""

    def __init__(
        self,
        api_key: str,
        *,
        client: anthropic.AsyncAnthropic | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key)
        self._sleep = sleep

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        messages = [_to_message_param(m) for m in request.messages]
        system = request.system if request.system is not None else anthropic.omit
        output_config = (
            _json_schema_output_config(request.output_schema)
            if request.output_schema is not None
            else anthropic.omit
        )
        response = await self._create_with_retry(
            model=request.model_id,
            max_tokens=request.max_tokens,
            system=system,
            messages=messages,
            output_config=output_config,
        )

        text = _first_text(response.content)
        usage = _usage(response.usage, request.model_id)
        stop_reason = response.stop_reason or "unknown"
        truncated = stop_reason == "max_tokens"
        parsed = None
        schema_error: str | None = None
        if request.output_schema is not None:
            try:
                parsed = request.output_schema.model_validate_json(text)
            except ValidationError as exc:
                # Say which of the two it actually was. A truncated response
                # reaches pydantic as an empty or half-finished string and is
                # reported as "Invalid JSON: EOF while parsing a value" - a
                # message that reads like the model wrote malformed JSON when
                # in fact it never finished writing. Live run `009ea59f-...`
                # lost two `repair_api` attempts and `build_api`'s whole
                # `cycle_budget` to that ambiguity.
                schema_error = (
                    f"response truncated at max_tokens={request.max_tokens} "
                    f"({usage.output_tokens} output tokens, {len(text)} characters of text "
                    f"emitted); the model did not finish its response. Underlying parse "
                    f"error: {exc}"
                    if truncated
                    else str(exc)
                )

        return CompletionResult(
            text=text,
            parsed=parsed,
            schema_error=schema_error,
            truncated=truncated,
            model_id=response.model,
            stop_reason=stop_reason,
            usage=usage,
        )

    async def _create_with_retry(self, **kwargs: Any) -> Any:
        """Bounded retry for transient connection failures only - never for
        `APIStatusError` (a real response from Anthropic, e.g. a 4xx/5xx),
        which fails immediately as before. `_MAX_ATTEMPTS` is small and fixed,
        matching `structured.py`'s "exactly one" bounded-repair philosophy:
        enough to ride out a network blip, never an unbounded/silent loop."""
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return await self._client.messages.create(**kwargs)
            except _TRANSIENT_ERRORS as exc:
                if attempt == _MAX_ATTEMPTS:
                    raise ProviderError(
                        f"could not reach Anthropic after {_MAX_ATTEMPTS} attempts: {exc}"
                    ) from exc
                await self._sleep(_RETRY_BACKOFF_SECONDS * attempt)
            except anthropic.APIStatusError as exc:
                raise ProviderError(f"Anthropic returned {exc.status_code}: {exc.message}") from exc
        raise AssertionError("unreachable: loop above always returns or raises")


def _enforce_additional_properties_false(schema: dict[str, Any]) -> None:
    """Anthropic's raw-schema structured-output mode requires every
    object-typed node to explicitly declare `additionalProperties: false` -
    a schema that merely omits the key (JSON Schema's own default, meaning
    "permissive") is rejected outright with a 400 before any completion is
    attempted, as opposed to a normal `CompletionResult.schema_error` this
    module's caller (`structured.py`) could otherwise repair.

    Pydantic only emits the key when a model sets `extra="forbid"`
    (`contracts.base.ArtifactModel` does, so every ordinary agent output
    schema is already fine) - a plain `BaseModel` output schema is not, and
    found live (`237d6873-...`, the first run ever to reach `migration`):
    `agents.migration._MigrationProposal` had no `extra="forbid"`, and
    `migration` has no `ON_FAILURE` recovery edge, so this one omission
    crashed the entire run outright. That specific model is now fixed at the
    source, but this function is the provider-boundary backstop for *any*
    Pydantic model or nested submodel used as `output_schema`, present or
    future, that makes the same omission - the LLM boundary is where the
    wire contract with a specific vendor's structured-output mode belongs,
    not scattered per-model reliance on remembering one config flag.

    Mutates `schema` in place; only fills in what pydantic left unset, never
    overrides an explicit value. Pydantic v2 flattens every nested
    `BaseModel` into a top-level `$defs` entry (never inline), so one pass
    over the root plus `$defs` covers a schema of any nesting depth.
    """
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
    for definition in schema.get("$defs", {}).values():
        if isinstance(definition, dict) and definition.get("type") == "object":
            definition.setdefault("additionalProperties", False)


def _json_schema_output_config(schema: type[Any]) -> OutputConfigParam:
    raw_schema = schema.model_json_schema()
    _enforce_additional_properties_false(raw_schema)
    return {"format": {"type": "json_schema", "schema": raw_schema}}
