"""Capability taxonomy and `ModelRouter` resolution (docs/02 Phase 2)."""

from __future__ import annotations

import pytest

from ases.providers.models import (
    MODEL_CATALOG,
    ModelCatalogEntryNotFoundError,
    ModelNeeds,
    spec_for,
)
from ases.providers.router import ModelRouter, NoModelSatisfiesNeedsError


def test_catalog_has_exactly_the_two_phase_2_models() -> None:
    assert {spec.id for spec in MODEL_CATALOG} == {"claude-opus-5", "claude-sonnet-5"}


def test_spec_for_unknown_model_raises() -> None:
    with pytest.raises(ModelCatalogEntryNotFoundError):
        spec_for("claude-nonexistent")


def test_high_reasoning_resolves_to_opus() -> None:
    router = ModelRouter()
    spec = router.resolve(ModelNeeds(reasoning="high"))
    assert spec.id == "claude-opus-5"


def test_low_reasoning_resolves_to_the_cheaper_model() -> None:
    router = ModelRouter()
    spec = router.resolve(ModelNeeds(reasoning="low"))
    assert spec.id == "claude-sonnet-5"


def test_medium_reasoning_does_not_pay_for_opus() -> None:
    router = ModelRouter()
    spec = router.resolve(ModelNeeds(reasoning="medium"))
    assert spec.id == "claude-sonnet-5"


def test_router_never_raises_for_needs_the_catalog_can_satisfy() -> None:
    router = ModelRouter()
    for reasoning in ("low", "medium", "high"):
        router.resolve(ModelNeeds(reasoning=reasoning))  # type: ignore[arg-type]


def test_no_model_satisfies_impossible_needs() -> None:
    router = ModelRouter(catalog=())
    with pytest.raises(NoModelSatisfiesNeedsError):
        router.resolve(ModelNeeds(reasoning="low"))


def test_cost_usd_matches_published_per_million_token_pricing() -> None:
    opus = spec_for("claude-opus-5")
    # 1,000,000 input + 1,000,000 output tokens at $5 / $25 per MTok.
    assert opus.cost_usd(input_tokens=1_000_000, output_tokens=1_000_000) == pytest.approx(30.0)


def test_cost_usd_is_zero_for_zero_usage() -> None:
    sonnet = spec_for("claude-sonnet-5")
    assert sonnet.cost_usd(input_tokens=0, output_tokens=0) == 0.0


def test_a_model_swap_does_not_change_reasoning_capability_ordering() -> None:
    """Regression guard for the router's core promise: raising `needs.reasoning`
    never resolves to a *cheaper* model than a lower request would."""
    router = ModelRouter()
    low = router.resolve(ModelNeeds(reasoning="low"))
    high = router.resolve(ModelNeeds(reasoning="high"))
    assert low.price_output_per_mtok <= high.price_output_per_mtok
