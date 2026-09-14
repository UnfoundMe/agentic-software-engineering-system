"""Structured-output validation with exactly one bounded repair attempt
(docs/02 Phase 2)."""

from __future__ import annotations

import pytest

from ases.contracts.artifacts import RequirementSpec
from ases.providers.base import (
    CompletionRequest,
    CompletionResult,
    TruncatedCompletionError,
)
from ases.providers.structured import StructuredOutputExhaustedError, complete_structured


def _request(**overrides: object) -> CompletionRequest:
    defaults: dict[str, object] = {
        "model_id": "claude-sonnet-5",
        "prompt_version": "p@v1",
        "rendered_prompt": "hello",
        "max_tokens": 100,
        "output_schema": RequirementSpec,
    }
    defaults.update(overrides)
    return CompletionRequest(**defaults)  # type: ignore[arg-type]


class _ScriptedProvider:
    """Returns one result per call, in order, and records every request it saw -
    a hand-rolled test double, not a mock library, since what matters here is
    the exact sequence and content of calls `complete_structured` makes."""

    def __init__(self, results: list[CompletionResult]) -> None:
        self._results = list(results)
        self.requests: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.requests.append(request)
        return self._results.pop(0)


def _ok(spec: RequirementSpec) -> CompletionResult:
    return CompletionResult(
        text=spec.model_dump_json(), parsed=spec, model_id="m", stop_reason="end_turn"
    )


def _failed(error: str) -> CompletionResult:
    return CompletionResult(
        text="not json", schema_error=error, model_id="m", stop_reason="end_turn"
    )


async def test_a_first_try_success_needs_no_repair_call() -> None:
    spec = RequirementSpec(summary="s", source_text="raw")
    provider = _ScriptedProvider([_ok(spec)])

    result = await complete_structured(provider, _request())

    assert result.parsed == spec
    assert len(provider.requests) == 1


async def test_a_failure_then_success_is_returned_after_exactly_one_repair() -> None:
    spec = RequirementSpec(summary="s", source_text="raw")
    provider = _ScriptedProvider([_failed("missing field 'summary'"), _ok(spec)])

    result = await complete_structured(provider, _request())

    assert result.parsed == spec
    assert len(provider.requests) == 2


async def test_the_repair_request_carries_the_original_error_message() -> None:
    provider = _ScriptedProvider(
        [_failed("missing field 'summary'"), _ok(RequirementSpec(summary="s", source_text="r"))]
    )
    await complete_structured(provider, _request())
    repair_prompt = provider.requests[1].rendered_prompt
    assert "missing field 'summary'" in repair_prompt


async def test_two_failures_raise_after_exactly_two_attempts_never_more() -> None:
    provider = _ScriptedProvider([_failed("bad json"), _failed("still bad json")])

    with pytest.raises(StructuredOutputExhaustedError) as excinfo:
        await complete_structured(provider, _request())

    assert len(provider.requests) == 2
    assert excinfo.value.schema_name == "RequirementSpec"
    assert "bad json" in excinfo.value.first_error
    assert "still bad json" in excinfo.value.second_error


async def test_requires_an_output_schema() -> None:
    provider = _ScriptedProvider([])
    with pytest.raises(ValueError, match="output_schema"):
        await complete_structured(provider, _request(output_schema=None))


# --- truncation is not a schema violation ----------------------------------
#
# Live run `009ea59f-...`: two `repair_*` nodes failed with byte-identical
# "Invalid JSON: EOF while parsing a value at line 1 column 0
# [input_value='']" errors. The model had not written bad JSON - it had
# written nothing, having spent the whole `max_tokens` ceiling inside its
# thinking block. The bounded repair then re-sent the same prompt under the
# same ceiling and truncated again at the same place, so the one repair
# attempt the boundary gets was spent without ever changing the condition.


def _truncated(text: str = "") -> CompletionResult:
    """What the Anthropic adapter returns for `stop_reason == "max_tokens"`."""
    return CompletionResult(
        text=text,
        schema_error="response truncated at max_tokens=100; the model did not finish",
        truncated=True,
        model_id="m",
        stop_reason="max_tokens",
    )


async def test_a_truncated_response_is_retried_with_more_room_not_the_same_ceiling() -> None:
    spec = RequirementSpec(summary="s", source_text="raw")
    provider = _ScriptedProvider([_truncated(), _ok(spec)])

    result = await complete_structured(provider, _request(max_tokens=100))

    assert result.parsed == spec
    first, retry = provider.requests
    assert first.max_tokens == 100
    assert retry.max_tokens == 200  # _TRUNCATION_HEADROOM
    # ...and told to be brief, rather than handed a validation error that
    # was never the real problem.
    assert "cut off before it finished" in retry.rendered_prompt
    assert "did not validate against" not in retry.rendered_prompt


async def test_a_response_that_truncates_twice_is_a_truncation_error_not_a_schema_error() -> None:
    provider = _ScriptedProvider([_truncated(), _truncated()])

    with pytest.raises(TruncatedCompletionError) as excinfo:
        await complete_structured(provider, _request(max_tokens=100))

    message = str(excinfo.value)
    assert "truncated at max_tokens" in message
    assert "agent protocol failure, not a source-code failure" in message
    # The misleading framing the live run produced must not reappear.
    assert "Invalid JSON" not in message


async def test_a_retry_that_stops_truncating_but_still_fails_is_a_schema_error() -> None:
    """More room fixed the truncation and revealed a genuine schema problem
    underneath. That is a different failure and must be reported as one."""
    provider = _ScriptedProvider([_truncated(), _failed("summary: field required")])

    with pytest.raises(StructuredOutputExhaustedError) as excinfo:
        await complete_structured(provider, _request(max_tokens=100))

    assert "summary: field required" in str(excinfo.value)


async def test_the_headroom_retry_is_capped_so_one_retry_cannot_become_a_blank_cheque() -> None:
    provider = _ScriptedProvider([_truncated(), _truncated()])

    with pytest.raises(TruncatedCompletionError):
        await complete_structured(provider, _request(max_tokens=30_000))

    assert provider.requests[1].max_tokens == 32_000  # _MAX_TOKENS_CEILING, not 60_000


async def test_an_ordinary_schema_failure_still_takes_the_repair_path_unchanged() -> None:
    """The pre-existing behaviour must be untouched: a model that answered
    under its ceiling, but wrongly, still gets the validation error fed back
    at the same `max_tokens`."""
    spec = RequirementSpec(summary="s", source_text="raw")
    provider = _ScriptedProvider([_failed("summary: field required"), _ok(spec)])

    result = await complete_structured(provider, _request(max_tokens=100))

    assert result.parsed == spec
    retry = provider.requests[1]
    assert retry.max_tokens == 100
    assert "did not validate against the RequirementSpec schema" in retry.rendered_prompt
