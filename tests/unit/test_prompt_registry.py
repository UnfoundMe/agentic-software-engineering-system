"""Versioned prompt templates (docs/02 Phase 2)."""

from __future__ import annotations

import pytest

from ases.providers.prompts.registry import (
    MissingPromptVariableError,
    PromptNotFoundError,
    PromptRegistry,
    PromptTemplate,
    required_variables,
)


def _registry() -> PromptRegistry:
    registry = PromptRegistry()
    registry.register(PromptTemplate(name="req_analysis", version=1, template="Analyze: {text}"))
    registry.register(PromptTemplate(name="req_analysis", version=2, template="v2 analyze: {text}"))
    return registry


def test_get_with_no_version_returns_the_highest_registered() -> None:
    template = _registry().get("req_analysis")
    assert template.version == 2


def test_get_with_an_explicit_version_is_pinned() -> None:
    template = _registry().get("req_analysis", version=1)
    assert template.template == "Analyze: {text}"


def test_unknown_prompt_name_raises() -> None:
    with pytest.raises(PromptNotFoundError):
        _registry().get("nonexistent")


def test_unknown_pinned_version_raises() -> None:
    with pytest.raises(PromptNotFoundError):
        _registry().get("req_analysis", version=99)


def test_render_substitutes_variables() -> None:
    rendered = _registry().get("req_analysis", version=1).render(text="hello")
    assert rendered == "Analyze: hello"


def test_render_missing_variable_raises_with_prompt_identity() -> None:
    template = _registry().get("req_analysis", version=1)
    with pytest.raises(MissingPromptVariableError) as excinfo:
        template.render()
    assert excinfo.value.prompt_version == "req_analysis@v1"
    assert excinfo.value.variable == "text"


def test_prompt_version_identifies_name_and_version() -> None:
    template = _registry().get("req_analysis", version=2)
    assert template.prompt_version == "req_analysis@v2"


def test_versions_lists_every_registered_version_sorted() -> None:
    assert _registry().versions("req_analysis") == (1, 2)


def test_versions_of_unknown_name_is_empty() -> None:
    assert _registry().versions("nonexistent") == ()


def test_required_variables_extracts_format_fields() -> None:
    assert required_variables("Hello {name}, you are {age} years old") == {"name", "age"}


def test_required_variables_of_a_literal_template_is_empty() -> None:
    assert required_variables("no placeholders here") == frozenset()


def test_same_wording_different_version_yields_different_prompt_version() -> None:
    """This is the property the cassette key relies on: bumping a template's
    version changes `prompt_version` even if nothing else about the call site
    changed, so a re-recorded cassette never silently mixes with the old one."""
    a = PromptTemplate(name="x", version=1, template="same text")
    b = PromptTemplate(name="x", version=2, template="same text")
    assert a.prompt_version != b.prompt_version
