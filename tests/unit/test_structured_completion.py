"""Structured-output validation with exactly one bounded repair attempt
(docs/02 Phase 2)."""

from __future__ import annotations

import pytest

from ases.contracts.artifacts import RequirementSpec
from ases.providers.base import CompletionRequest, CompletionResult
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
