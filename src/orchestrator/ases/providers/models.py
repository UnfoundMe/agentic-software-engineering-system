"""Model capability taxonomy and the fixed catalog `ModelRouter` resolves
against.

Agents declare *needs* (`ModelNeeds`); they never name a model id. This module
is the only place a model id is allowed to appear as a literal string outside
`anthropic_provider.py`'s tests - see docs/02 section 0's locked decision
("Model selection: Capability-based `ModelRouter`; agents never name a
model").

Pricing is per-million-tokens, current Anthropic first-party API rates for
the two models docs/02 Phase 2 names (`claude-opus-5`, `claude-sonnet-5`).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

ReasoningLevel = Literal["low", "medium", "high"]
ContextSize = Literal["small", "large"]
LatencyClass = Literal["interactive", "batch"]
CostClass = Literal["low", "medium", "high"]

#: Total ordering used by the router to find the cheapest model that still
#: meets a requested reasoning level.
_REASONING_RANK: dict[ReasoningLevel, int] = {"low": 0, "medium": 1, "high": 2}


class ModelNeeds(BaseModel):
    """What an agent asks for. Never a model id - see module docstring."""

    model_config = ConfigDict(frozen=True)

    reasoning: ReasoningLevel
    context: ContextSize = "small"
    structured_output: bool = True
    latency: LatencyClass = "batch"


class ModelSpec(BaseModel):
    """One entry in the catalog: what a model can do, and what it costs."""

    model_config = ConfigDict(frozen=True)

    id: str
    max_reasoning: ReasoningLevel
    cost_class: CostClass
    price_input_per_mtok: float
    price_output_per_mtok: float
    supports_structured_output: bool = True
    #: Both catalog models currently ship a 1M-token window; kept as a field
    #: (rather than assumed) so a future smaller-context model is representable.
    max_context_tokens: int = 1_000_000

    @property
    def reasoning_rank(self) -> int:
        return _REASONING_RANK[self.max_reasoning]

    def cost_usd(self, *, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.price_input_per_mtok + output_tokens * self.price_output_per_mtok
        ) / 1_000_000


#: The Phase 2 catalog: exactly the two models docs/02 Phase 2 names.
#: Extending this to a third model is a one-line addition, never a code
#: change to `ModelRouter` - see docs/05 section 11's extension-point table.
MODEL_CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="claude-sonnet-5",
        max_reasoning="medium",
        cost_class="low",
        price_input_per_mtok=2.00,
        price_output_per_mtok=10.00,
    ),
    ModelSpec(
        id="claude-opus-5",
        max_reasoning="high",
        cost_class="high",
        price_input_per_mtok=5.00,
        price_output_per_mtok=25.00,
    ),
)


class ModelCatalogEntryNotFoundError(KeyError):
    """No catalog entry has this model id."""


def spec_for(model_id: str, catalog: tuple[ModelSpec, ...] = MODEL_CATALOG) -> ModelSpec:
    for spec in catalog:
        if spec.id == model_id:
            return spec
    raise ModelCatalogEntryNotFoundError(model_id)


def reasoning_rank(level: ReasoningLevel) -> int:
    """Public accessor for `_REASONING_RANK`, so callers outside this module
    (`router.py`) never need to reach into a private table directly."""
    return _REASONING_RANK[level]


__all__ = [
    "MODEL_CATALOG",
    "ContextSize",
    "CostClass",
    "LatencyClass",
    "ModelCatalogEntryNotFoundError",
    "ModelNeeds",
    "ModelSpec",
    "ReasoningLevel",
    "reasoning_rank",
    "spec_for",
]
