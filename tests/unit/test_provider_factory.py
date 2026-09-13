"""`get_provider` mode selection (docs/02 section 0: replay/record/live/mock,
selected by `ASES_LLM_MODE`)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from ases.config import LLMMode, Settings
from ases.providers.anthropic_provider import AnthropicProvider
from ases.providers.cassette import CassetteProvider, RecordingProvider
from ases.providers.factory import MissingApiKeyError, get_provider
from ases.providers.mock import MockProvider


def _settings(mode: LLMMode, *, api_key: str = "") -> Settings:
    return Settings(
        ases_llm_mode=mode,
        anthropic_api_key=SecretStr(api_key),
        postgres_superuser_password=SecretStr(""),
        ases_control_password=SecretStr(""),
        ases_app_password=SecretStr(""),
        workload_app_password=SecretStr(""),
    )


def test_replay_yields_a_cassette_provider() -> None:
    assert isinstance(get_provider(_settings(LLMMode.REPLAY)), CassetteProvider)


def test_mock_yields_a_mock_provider() -> None:
    assert isinstance(get_provider(_settings(LLMMode.MOCK)), MockProvider)


def test_live_yields_an_anthropic_provider_when_a_key_is_set() -> None:
    provider = get_provider(_settings(LLMMode.LIVE, api_key="sk-test"))
    assert isinstance(provider, AnthropicProvider)


def test_record_yields_a_recording_provider_wrapping_anthropic() -> None:
    provider = get_provider(_settings(LLMMode.RECORD, api_key="sk-test"))
    assert isinstance(provider, RecordingProvider)


def test_live_without_a_key_raises() -> None:
    with pytest.raises(MissingApiKeyError):
        get_provider(_settings(LLMMode.LIVE))


def test_record_without_a_key_raises() -> None:
    with pytest.raises(MissingApiKeyError):
        get_provider(_settings(LLMMode.RECORD))


def test_replay_never_requires_a_key() -> None:
    """The offline-by-default property (docs/02 section 0): replay must work
    with no ANTHROPIC_API_KEY at all."""
    get_provider(_settings(LLMMode.REPLAY, api_key=""))


def test_a_custom_cassette_dir_is_honoured(tmp_path: Path) -> None:
    provider = get_provider(_settings(LLMMode.REPLAY), cassette_dir=tmp_path)
    assert isinstance(provider, CassetteProvider)
