"""Versioned prompt registry.

Every prompt an agent renders is registered here under a `(name, version)`
pair. Versioning exists because `prompt_version` is half of the cassette key
(`providers.cassette.cassette_key`): changing a template's wording without
bumping its version would silently mix old and new prompts under one
cassette, which is exactly the kind of drift the versioning is meant to make
visible instead of silent.

Templates are plain `str.format`-style strings (`{field}` placeholders) - no
templating engine, because the prompts registered here (Phase 4 onward) are
short, reviewable instructions, not documents with control flow.

No agent exists yet to register a real prompt (Phase 4). This module is the
mechanism docs/02 Phase 2 asks for; the first real entries land with the
first agent.
"""

from __future__ import annotations

from string import Formatter

from pydantic import BaseModel, ConfigDict


class PromptTemplate(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: int
    template: str

    @property
    def prompt_version(self) -> str:
        """Identifier embedded in the cassette key - see module docstring."""
        return f"{self.name}@v{self.version}"

    def render(self, **variables: str) -> str:
        try:
            return self.template.format(**variables)
        except KeyError as exc:
            raise MissingPromptVariableError(self.prompt_version, exc.args[0]) from exc


class PromptNotFoundError(KeyError):
    """No template is registered under this name (and version, if given)."""


class MissingPromptVariableError(KeyError):
    """`render()` was called without a variable the template requires."""

    def __init__(self, prompt_version: str, variable: str) -> None:
        super().__init__(f"{prompt_version} requires variable {variable!r}")
        self.prompt_version = prompt_version
        self.variable = variable


class PromptRegistry:
    """In-process registry of `PromptTemplate`s, keyed by name and version.

    `get(name)` with no version returns the highest registered version - the
    common case, an agent always wants the current prompt. A pinned version is
    for reproducing a past cassette exactly, or for a repair pass that must
    stay on the version whose output it is repairing.
    """

    def __init__(self) -> None:
        self._templates: dict[str, dict[int, PromptTemplate]] = {}

    def register(self, template: PromptTemplate) -> None:
        by_version = self._templates.setdefault(template.name, {})
        by_version[template.version] = template

    def get(self, name: str, version: int | None = None) -> PromptTemplate:
        by_version = self._templates.get(name)
        if not by_version:
            raise PromptNotFoundError(name)
        if version is None:
            return by_version[max(by_version)]
        try:
            return by_version[version]
        except KeyError:
            raise PromptNotFoundError(f"{name}@v{version}") from None

    def versions(self, name: str) -> tuple[int, ...]:
        return tuple(sorted(self._templates.get(name, {})))


def required_variables(template: str) -> frozenset[str]:
    """Field names a `str.format` template references - useful for a caller
    that wants to validate inputs before rendering rather than catching
    `MissingPromptVariableError` after the fact."""
    return frozenset(field for _, field, _, _ in Formatter().parse(template) if field)
