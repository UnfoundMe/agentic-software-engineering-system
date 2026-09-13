"""Deterministic, offline replay of previously recorded LLM completions.

`ASES_LLM_MODE=replay` is the default everywhere, including CI (docs/02
section 0): a run must reproduce identically with no network access and no
API key. The mechanism is a content-addressed cassette store keyed by

    sha256(prompt_version + rendered_prompt + output_schema)

deliberately **excluding the model id** - docs/02 Phase 2: "a model swap in
the router does not invalidate cassettes." `output_schema` is folded in via
its JSON schema (not just the class name), so changing a contract's fields
also changes the key rather than silently replaying a stale shape against a
new one.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ases.kernel.hashing import digest_text
from ases.providers.base import (
    CompletionRequest,
    CompletionResult,
    CompletionUsage,
    LLMProvider,
)

DEFAULT_CASSETTE_DIR = Path(__file__).parent / "cassettes"


def _schema_fingerprint(request: CompletionRequest) -> str:
    if request.output_schema is None:
        return ""
    return json.dumps(request.output_schema.model_json_schema(), sort_keys=True)


def cassette_key(request: CompletionRequest) -> str:
    """The cassette identity. See module docstring for what is (and is not)
    included, and why."""
    return digest_text(
        "\x1e".join((request.prompt_version, request.rendered_prompt, _schema_fingerprint(request)))
    )


class CassetteMissError(LookupError):
    """No recorded cassette matches this request's key.

    Raised rather than falling back to a live call: `replay` mode's whole
    point is that it never reaches the network, even by accident.
    """

    def __init__(self, key: str, request: CompletionRequest) -> None:
        super().__init__(
            f"no cassette for key {key} (prompt_version={request.prompt_version!r}). "
            "Record one first with ASES_LLM_MODE=record."
        )
        self.key = key


class Cassette(BaseModel):
    """One recorded completion, persisted as JSON."""

    model_config = ConfigDict(frozen=True)

    key: str
    prompt_version: str
    model_id: str  # informational only; never part of `key`
    text: str
    parsed_kind: str | None = None
    parsed: dict[str, object] | None = None
    stop_reason: str
    usage: CompletionUsage


class CassetteStore:
    """Reads and writes cassette files, one JSON document per key."""

    def __init__(self, directory: Path = DEFAULT_CASSETTE_DIR) -> None:
        self._dir = directory

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def load(self, key: str) -> Cassette | None:
        path = self._path(key)
        if not path.exists():
            return None
        return Cassette.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, cassette: Cassette) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path(cassette.key).write_text(cassette.model_dump_json(indent=2), encoding="utf-8")


class CassetteProvider:
    """`LLMProvider` that only ever replays. Never touches the network."""

    def __init__(self, store: CassetteStore | None = None) -> None:
        self._store = store or CassetteStore()

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        key = cassette_key(request)
        cassette = self._store.load(key)
        if cassette is None:
            raise CassetteMissError(key, request)
        return _result_from_cassette(cassette, request)


class RecordingProvider:
    """`LLMProvider` that calls a real provider once and saves what it said.

    Wraps any other `LLMProvider` (in practice, `AnthropicProvider`) - the
    recording behaviour is a decorator, not a property of the Anthropic
    adapter itself, so recording against a future second vendor needs no new
    code here.
    """

    def __init__(self, inner: LLMProvider, store: CassetteStore | None = None) -> None:
        self._inner = inner
        self._store = store or CassetteStore()

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        result = await self._inner.complete(request)
        self._store.save(_cassette_from_result(cassette_key(request), request, result))
        return result


def _cassette_from_result(
    key: str, request: CompletionRequest, result: CompletionResult
) -> Cassette:
    return Cassette(
        key=key,
        prompt_version=request.prompt_version,
        model_id=result.model_id,
        text=result.text,
        parsed_kind=type(result.parsed).__name__ if result.parsed is not None else None,
        parsed=result.parsed.model_dump(mode="json") if result.parsed is not None else None,
        stop_reason=result.stop_reason,
        usage=result.usage,
    )


def _result_from_cassette(cassette: Cassette, request: CompletionRequest) -> CompletionResult:
    parsed: BaseModel | None = None
    if cassette.parsed is not None and request.output_schema is not None:
        parsed = request.output_schema.model_validate(cassette.parsed)
    return CompletionResult(
        text=cassette.text,
        parsed=parsed,
        model_id=request.model_id,  # the run's current model, not the recording's
        stop_reason=cassette.stop_reason,
        usage=cassette.usage,
    )


__all__ = [
    "Cassette",
    "CassetteMissError",
    "CassetteProvider",
    "CassetteStore",
    "RecordingProvider",
    "cassette_key",
]
