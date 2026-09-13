"""Cassette record/replay (docs/02 Phase 2 exit criterion: "the same run
replays identically twice offline; a model swap in the router does not
invalidate cassettes")."""

from __future__ import annotations

from pathlib import Path

import pytest

from ases.contracts.artifacts import RequirementSpec
from ases.providers.base import CompletionRequest, CompletionResult, CompletionUsage
from ases.providers.cassette import (
    CassetteMissError,
    CassetteProvider,
    CassetteStore,
    RecordingProvider,
    cassette_key,
)


def _request(**overrides: object) -> CompletionRequest:
    defaults: dict[str, object] = {
        "model_id": "claude-sonnet-5",
        "prompt_version": "p@v1",
        "rendered_prompt": "hello world",
        "max_tokens": 100,
    }
    defaults.update(overrides)
    return CompletionRequest(**defaults)  # type: ignore[arg-type]


class _FakeInner:
    """Stands in for a real provider (Anthropic) during recording, so this
    test never touches the network."""

    def __init__(self, result: CompletionResult) -> None:
        self._result = result
        self.calls = 0

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.calls += 1
        return self._result


def test_key_excludes_the_model_id() -> None:
    """The load-bearing property: swapping models must not invalidate a
    recorded cassette."""
    sonnet = _request(model_id="claude-sonnet-5")
    opus = _request(model_id="claude-opus-5")
    assert cassette_key(sonnet) == cassette_key(opus)


def test_key_changes_when_the_prompt_version_changes() -> None:
    a = _request(prompt_version="p@v1")
    b = _request(prompt_version="p@v2")
    assert cassette_key(a) != cassette_key(b)


def test_key_changes_when_the_rendered_prompt_changes() -> None:
    a = _request(rendered_prompt="hello")
    b = _request(rendered_prompt="goodbye")
    assert cassette_key(a) != cassette_key(b)


def test_key_changes_when_the_output_schema_changes() -> None:
    from ases.contracts.artifacts import DesignSpec

    a = _request(output_schema=RequirementSpec)
    b = _request(output_schema=DesignSpec)
    assert cassette_key(a) != cassette_key(b)


def test_key_is_deterministic_across_calls() -> None:
    request = _request()
    assert cassette_key(request) == cassette_key(request)


async def test_replay_without_a_recording_raises_cassette_miss(tmp_path: Path) -> None:
    provider = CassetteProvider(CassetteStore(tmp_path))
    with pytest.raises(CassetteMissError):
        await provider.complete(_request())


async def test_record_then_replay_yields_the_same_result(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path)
    inner = _FakeInner(
        CompletionResult(
            text="hello back",
            model_id="claude-sonnet-5",
            stop_reason="end_turn",
            usage=CompletionUsage(input_tokens=3, output_tokens=4, usd=0.001),
        )
    )
    recorder = RecordingProvider(inner, store)
    recorded = await recorder.complete(_request())

    replayer = CassetteProvider(store)
    replayed = await replayer.complete(_request())

    assert replayed.text == recorded.text == "hello back"
    assert replayed.usage == recorded.usage
    assert inner.calls == 1  # replay never touched the "network"


async def test_replaying_twice_is_byte_identical(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path)
    inner = _FakeInner(
        CompletionResult(text="deterministic", model_id="claude-sonnet-5", stop_reason="end_turn")
    )
    request = _request(prompt_version="test_replay_twice@v1")
    await RecordingProvider(inner, store).complete(request)

    provider = CassetteProvider(store)
    first = await provider.complete(request)
    second = await provider.complete(request)
    assert first == second


async def test_a_model_swap_replays_from_the_same_cassette(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path)
    inner = _FakeInner(
        CompletionResult(text="opus said this", model_id="claude-opus-5", stop_reason="end_turn")
    )
    await RecordingProvider(inner, store).complete(_request(model_id="claude-opus-5"))

    # A later run resolves a *different* model for the same prompt (e.g. the
    # router's policy changed) - replay must still find the cassette.
    replayer = CassetteProvider(store)
    result = await replayer.complete(_request(model_id="claude-sonnet-5"))
    assert result.text == "opus said this"


async def test_replayed_structured_output_is_revalidated_against_the_current_schema(
    tmp_path: Path,
) -> None:
    store = CassetteStore(tmp_path)
    spec = RequirementSpec(summary="s", source_text="raw")
    inner = _FakeInner(
        CompletionResult(
            text=spec.model_dump_json(), parsed=spec, model_id="m", stop_reason="end_turn"
        )
    )
    request = _request(output_schema=RequirementSpec)
    await RecordingProvider(inner, store).complete(request)

    result = await CassetteProvider(store).complete(request)
    assert result.parsed == spec
    assert isinstance(result.parsed, RequirementSpec)
