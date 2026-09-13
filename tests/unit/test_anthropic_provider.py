"""`AnthropicProvider` (docs/02 Phase 2: the live LLM boundary adapter).

Every test injects a fake client in place of `anthropic.AsyncAnthropic`, so
none of this touches the network - see docs/06's "never claim success without
actual evidence": these tests prove the request/response *mapping* is
correct, not that the live Anthropic API behaves as documented (that would
need ASES_LLM_MODE=record against a real key, which is deliberately outside
the default, offline test suite).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from ases.contracts.artifacts import CodePatch, RequirementSpec
from ases.providers.anthropic_provider import AnthropicProvider
from ases.providers.base import CompletionRequest, ProviderError


def _request(**overrides: object) -> CompletionRequest:
    defaults: dict[str, object] = {
        "model_id": "claude-sonnet-5",
        "prompt_version": "p@v1",
        "rendered_prompt": "hello",
        "max_tokens": 100,
    }
    defaults.update(overrides)
    return CompletionRequest(**defaults)  # type: ignore[arg-type]


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class _FakeMessage:
    content: list[_FakeTextBlock]
    model: str
    stop_reason: str | None
    usage: _FakeUsage


class _FakeMessagesResource:
    def __init__(
        self, response: _FakeMessage | None = None, error: Exception | None = None
    ) -> None:
        self._response = response
        self._error = error
        self.last_kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> _FakeMessage:
        self.last_kwargs = kwargs
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


@dataclass
class _FakeClient:
    messages: _FakeMessagesResource = field(default_factory=_FakeMessagesResource)


def _provider(
    response: _FakeMessage | None = None, error: Exception | None = None
) -> tuple[AnthropicProvider, _FakeMessagesResource]:
    messages = _FakeMessagesResource(response, error)
    client = _FakeClient(messages=messages)
    return AnthropicProvider("test-key", client=client), messages  # type: ignore[arg-type]


async def test_plain_completion_extracts_text_and_usage() -> None:
    response = _FakeMessage(
        content=[_FakeTextBlock(text="hi there")],
        model="claude-sonnet-5",
        stop_reason="end_turn",
        usage=_FakeUsage(input_tokens=100, output_tokens=50),
    )
    provider, _ = _provider(response)

    result = await provider.complete(_request())

    assert result.text == "hi there"
    assert result.stop_reason == "end_turn"
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 50


async def test_cost_is_computed_from_the_catalog_price() -> None:
    response = _FakeMessage(
        content=[_FakeTextBlock(text="hi")],
        model="claude-sonnet-5",
        stop_reason="end_turn",
        usage=_FakeUsage(input_tokens=1_000_000, output_tokens=1_000_000),
    )
    provider, _ = _provider(response)
    result = await provider.complete(_request(model_id="claude-sonnet-5"))
    assert result.usage.usd == pytest.approx(12.0)  # $2 + $10 per MTok


async def test_a_model_outside_the_catalog_still_reports_token_usage() -> None:
    response = _FakeMessage(
        content=[_FakeTextBlock(text="hi")],
        model="some-future-model",
        stop_reason="end_turn",
        usage=_FakeUsage(input_tokens=10, output_tokens=5),
    )
    provider, _ = _provider(response)
    result = await provider.complete(_request(model_id="some-future-model"))
    assert result.usage.input_tokens == 10
    assert result.usage.usd == 0.0


async def test_no_text_block_yields_empty_text_rather_than_raising() -> None:
    response = _FakeMessage(
        content=[], model="claude-sonnet-5", stop_reason="end_turn", usage=_FakeUsage(0, 0)
    )
    provider, _ = _provider(response)
    result = await provider.complete(_request())
    assert result.text == ""


async def test_structured_request_sends_a_json_schema_output_config() -> None:
    response = _FakeMessage(
        content=[_FakeTextBlock(text='{"summary": "s", "source_text": "raw"}')],
        model="claude-sonnet-5",
        stop_reason="end_turn",
        usage=_FakeUsage(input_tokens=10, output_tokens=10),
    )
    provider, messages = _provider(response)
    await provider.complete(_request(output_schema=RequirementSpec))
    sent = messages.last_kwargs["output_config"]
    assert sent["format"]["type"] == "json_schema"
    assert sent["format"]["schema"]["properties"]["summary"]["type"] == "string"


class _PlainSchemaWithoutExtraForbid(BaseModel):
    """Stands in for `agents.migration._MigrationProposal`'s real defect
    shape: a plain `BaseModel` output schema with no `extra="forbid"`, so
    `model_json_schema()` omits `additionalProperties` entirely."""

    name: str


async def test_a_schema_missing_additional_properties_gets_it_filled_in() -> None:
    """Regression test for run `237d6873-...`: `migration` was the first
    node in any live run to use a non-`ArtifactModel` output schema
    (`_MigrationProposal`, no `extra="forbid"`), and Anthropic's raw-schema
    structured-output mode rejects a schema that merely omits
    `additionalProperties` with a 400 - before any completion is attempted,
    and with no `ON_FAILURE` recovery edge on `migration`, that crashed the
    entire run. The provider must backstop this regardless of which model is
    used, not rely on every schema author remembering the config flag."""
    response = _FakeMessage(
        content=[_FakeTextBlock(text='{"name": "x"}')],
        model="claude-sonnet-5",
        stop_reason="end_turn",
        usage=_FakeUsage(input_tokens=10, output_tokens=10),
    )
    provider, messages = _provider(response)
    await provider.complete(_request(output_schema=_PlainSchemaWithoutExtraForbid))
    sent_schema = messages.last_kwargs["output_config"]["format"]["schema"]
    assert sent_schema["additionalProperties"] is False


async def test_an_already_compliant_schema_is_left_alone() -> None:
    """`ArtifactModel` (`extra="forbid"`) already emits `additionalProperties:
    false` at every level - the backstop must not double-set or otherwise
    disturb a schema that is already correct, at the root or in `$defs`."""
    response = _FakeMessage(
        content=[_FakeTextBlock(text='{"summary": "s", "files": [], "schema_version": 1}')],
        model="claude-sonnet-5",
        stop_reason="end_turn",
        usage=_FakeUsage(input_tokens=10, output_tokens=10),
    )
    provider, messages = _provider(response)
    await provider.complete(_request(output_schema=CodePatch))
    sent_schema = messages.last_kwargs["output_config"]["format"]["schema"]
    assert sent_schema["additionalProperties"] is False
    for definition in sent_schema.get("$defs", {}).values():
        if definition.get("type") == "object":
            assert definition["additionalProperties"] is False


async def test_valid_structured_output_parses_successfully() -> None:
    response = _FakeMessage(
        content=[_FakeTextBlock(text='{"summary": "s", "source_text": "raw"}')],
        model="claude-sonnet-5",
        stop_reason="end_turn",
        usage=_FakeUsage(input_tokens=10, output_tokens=10),
    )
    provider, _ = _provider(response)
    result = await provider.complete(_request(output_schema=RequirementSpec))
    assert result.schema_error is None
    assert result.parsed == RequirementSpec(summary="s", source_text="raw")


async def test_invalid_structured_output_sets_schema_error_but_keeps_usage() -> None:
    """The critical property: a schema failure is data, not a swallowed
    exception - and usage is still reported, since the call was still billed."""
    response = _FakeMessage(
        content=[_FakeTextBlock(text="not even json")],
        model="claude-sonnet-5",
        stop_reason="end_turn",
        usage=_FakeUsage(input_tokens=10, output_tokens=10),
    )
    provider, _ = _provider(response)
    result = await provider.complete(_request(output_schema=RequirementSpec))
    assert result.parsed is None
    assert result.schema_error is not None
    assert result.usage.input_tokens == 10


async def test_missing_required_field_sets_schema_error() -> None:
    response = _FakeMessage(
        content=[_FakeTextBlock(text='{"summary": "s"}')],  # source_text is required
        model="claude-sonnet-5",
        stop_reason="end_turn",
        usage=_FakeUsage(input_tokens=10, output_tokens=10),
    )
    provider, _ = _provider(response)
    result = await provider.complete(_request(output_schema=RequirementSpec))
    assert result.parsed is None
    assert "source_text" in (result.schema_error or "")


async def test_a_connection_error_is_wrapped_as_provider_error() -> None:
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    provider, _ = _provider(error=anthropic.APIConnectionError(request=req))
    with pytest.raises(ProviderError):
        await provider.complete(_request())


async def test_an_api_status_error_is_wrapped_as_provider_error() -> None:
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx2.Response(500, request=req)
    error = anthropic.APIStatusError("server exploded", response=resp, body=None)
    provider, _ = _provider(error=error)
    with pytest.raises(ProviderError):
        await provider.complete(_request())


async def test_system_prompt_is_forwarded_when_present() -> None:
    response = _FakeMessage(
        content=[_FakeTextBlock(text="ok")],
        model="claude-sonnet-5",
        stop_reason="end_turn",
        usage=_FakeUsage(0, 0),
    )
    provider, messages = _provider(response)
    await provider.complete(_request(system="be terse"))
    assert messages.last_kwargs["system"] == "be terse"


async def test_history_is_forwarded_before_the_rendered_prompt() -> None:
    from ases.providers.base import Message

    response = _FakeMessage(
        content=[_FakeTextBlock(text="ok")],
        model="claude-sonnet-5",
        stop_reason="end_turn",
        usage=_FakeUsage(0, 0),
    )
    provider, messages = _provider(response)
    history = (
        Message(role="user", content="first turn"),
        Message(role="assistant", content="ack"),
    )
    await provider.complete(_request(rendered_prompt="second turn", history=history))
    sent = messages.last_kwargs["messages"]
    assert [m["content"] for m in sent] == ["first turn", "ack", "second turn"]
