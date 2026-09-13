"""`MockProvider` (docs/02 Phase 2: mock provider for unit tests)."""

from __future__ import annotations

import pytest

from ases.contracts.artifacts import RequirementSpec
from ases.providers.base import CompletionRequest, CompletionResult, CompletionUsage
from ases.providers.mock import MockProvider, NoScriptedResponseError


def _request(**overrides: object) -> CompletionRequest:
    defaults: dict[str, object] = {
        "model_id": "claude-sonnet-5",
        "prompt_version": "p@v1",
        "rendered_prompt": "hello",
        "max_tokens": 100,
    }
    defaults.update(overrides)
    return CompletionRequest(**defaults)  # type: ignore[arg-type]


async def test_default_echo_is_deterministic_on_prompt_length() -> None:
    provider = MockProvider()
    result = await provider.complete(_request(rendered_prompt="12345"))
    assert result.text == "mock:5"


async def test_default_echo_carries_the_requested_model_id() -> None:
    provider = MockProvider()
    result = await provider.complete(_request(model_id="claude-opus-5"))
    assert result.model_id == "claude-opus-5"


async def test_scripted_responses_are_consumed_in_order() -> None:
    provider = MockProvider()
    first = CompletionResult(text="first", model_id="m", stop_reason="end_turn")
    second = CompletionResult(text="second", model_id="m", stop_reason="end_turn")
    provider.respond_with(first)
    provider.respond_with(second)

    assert (await provider.complete(_request())).text == "first"
    assert (await provider.complete(_request())).text == "second"


async def test_after_the_queue_is_exhausted_it_falls_back_to_the_default_echo() -> None:
    provider = MockProvider()
    provider.respond_with(CompletionResult(text="only", model_id="m", stop_reason="end_turn"))
    await provider.complete(_request())
    result = await provider.complete(_request(rendered_prompt="ab"))
    assert result.text == "mock:2"


async def test_structured_request_with_no_scripted_response_raises_rather_than_guessing() -> None:
    """MockProvider must never invent field values for an arbitrary schema -
    see the module's docstring on why."""
    provider = MockProvider()
    with pytest.raises(NoScriptedResponseError):
        await provider.complete(_request(output_schema=RequirementSpec))


async def test_a_scripted_parsed_response_round_trips() -> None:
    provider = MockProvider()
    spec = RequirementSpec(summary="s", source_text="raw")
    provider.respond_with(
        CompletionResult(
            text=spec.model_dump_json(),
            parsed=spec,
            model_id="m",
            stop_reason="end_turn",
            usage=CompletionUsage(input_tokens=10, output_tokens=5, usd=0.01),
        )
    )
    result = await provider.complete(_request(output_schema=RequirementSpec))
    assert result.parsed == spec
    assert result.usage.input_tokens == 10
