"""Capability-based model selection.

Agents declare `ModelNeeds`; `ModelRouter.resolve` picks the cheapest catalog
entry that satisfies them. Centralising this in one file is what turns "use a
bigger model for architecture, a smaller one for boilerplate" into a policy
change here rather than a scattered set of model-id literals across agents
(docs/02 section 2, `providers/router.py`).
"""

from __future__ import annotations

from collections.abc import Sequence

from ases.providers.models import MODEL_CATALOG, ModelNeeds, ModelSpec, reasoning_rank


class NoModelSatisfiesNeedsError(RuntimeError):
    """No catalog entry meets the requested capabilities.

    Raised rather than silently downgrading to the closest match - an agent
    that needs high reasoning and gets a medium-reasoning model without
    anyone deciding that is an outcome nobody chose.
    """

    def __init__(self, needs: ModelNeeds) -> None:
        super().__init__(f"no catalog entry satisfies {needs!r}")
        self.needs = needs


class ModelRouter:
    """Resolves `ModelNeeds` to a `ModelSpec`. Stateless; safe to share."""

    def __init__(self, catalog: Sequence[ModelSpec] = MODEL_CATALOG) -> None:
        self._catalog = tuple(catalog)

    def resolve(self, needs: ModelNeeds) -> ModelSpec:
        candidates = [
            spec
            for spec in self._catalog
            if spec.reasoning_rank >= reasoning_rank(needs.reasoning)
            and spec.max_context_tokens >= _context_floor(needs)
            and (not needs.structured_output or spec.supports_structured_output)
        ]
        if not candidates:
            raise NoModelSatisfiesNeedsError(needs)
        # Cheapest candidate wins: reasoning/context are floors already
        # applied above, so among those that qualify, cost is the only
        # remaining axis to optimise.
        return min(candidates, key=lambda spec: spec.price_output_per_mtok)


def _context_floor(needs: ModelNeeds) -> int:
    # Both catalog entries currently ship a 1M window, so "large" and "small"
    # do not yet differentiate anything - the floor exists so a future
    # smaller-context model is correctly excluded from "large" requests
    # without a change to this function.
    return 200_000 if needs.context == "large" else 0
