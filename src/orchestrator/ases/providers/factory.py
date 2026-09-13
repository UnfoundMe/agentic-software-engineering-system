"""Selects an `LLMProvider` from `Settings.ases_llm_mode`.

The one place `ASES_LLM_MODE` is read to decide *which* provider exists.
Everything downstream (agents, once Phase 4 builds them) depends only on the
`LLMProvider` protocol, never on this function's result being any one
concrete type - docs/02 section 0: "replay (default, offline, deterministic),
record, live, mock."
"""

from __future__ import annotations

from pathlib import Path

from ases.config import LLMMode, Settings
from ases.providers.anthropic_provider import AnthropicProvider
from ases.providers.base import LLMProvider
from ases.providers.cassette import CassetteProvider, CassetteStore, RecordingProvider
from ases.providers.mock import MockProvider


class MissingApiKeyError(RuntimeError):
    """`record` or `live` mode was selected with no `ANTHROPIC_API_KEY` set."""


def get_provider(settings: Settings, *, cassette_dir: Path | None = None) -> LLMProvider:
    store = CassetteStore(cassette_dir) if cassette_dir is not None else CassetteStore()

    match settings.ases_llm_mode:
        case LLMMode.REPLAY:
            return CassetteProvider(store)
        case LLMMode.MOCK:
            return MockProvider()
        case LLMMode.RECORD:
            return RecordingProvider(_anthropic(settings), store)
        case LLMMode.LIVE:
            return _anthropic(settings)
    raise AssertionError(f"unhandled LLMMode {settings.ases_llm_mode}")  # pragma: no cover


def _anthropic(settings: Settings) -> AnthropicProvider:
    key = settings.anthropic_api_key.get_secret_value()
    if not key:
        raise MissingApiKeyError(
            f"ASES_LLM_MODE={settings.ases_llm_mode} requires ANTHROPIC_API_KEY to be set"
        )
    return AnthropicProvider(key)
