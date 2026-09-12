"""BudgetEntryGate: the one deterministic, agent-free gate built so far."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from ases.kernel.gates import BudgetEntryGate, BudgetLimits, GateVerdict
from ases.kernel.state import RunState, Usage

GENEROUS = BudgetLimits(max_tokens=1000, max_usd=10.0, max_wallclock_seconds=3600)


def _state(**usage_kwargs: object) -> RunState:
    return RunState(run_id=uuid4(), usage=Usage(**usage_kwargs), created_at=datetime.now(UTC))  # type: ignore[arg-type]


def test_passes_within_budget() -> None:
    result = BudgetEntryGate(GENEROUS).evaluate(_state())
    assert result.verdict is GateVerdict.PASS


def test_fails_over_token_budget() -> None:
    result = BudgetEntryGate(GENEROUS).evaluate(_state(input_tokens=2000))
    assert result.verdict is GateVerdict.FAIL
    assert "token budget" in result.reason


def test_fails_over_cost_budget() -> None:
    result = BudgetEntryGate(GENEROUS).evaluate(_state(usd=20.0))
    assert result.verdict is GateVerdict.FAIL
    assert "cost budget" in result.reason


def test_fails_over_wallclock_budget() -> None:
    tight = BudgetLimits(max_tokens=1000, max_usd=10.0, max_wallclock_seconds=1.0)
    state = RunState(run_id=uuid4(), created_at=datetime.now(UTC) - timedelta(seconds=10))
    result = BudgetEntryGate(tight).evaluate(state)
    assert result.verdict is GateVerdict.FAIL
    assert "wall-clock budget" in result.reason


def test_no_created_at_skips_wallclock_check() -> None:
    """A run with no created_at yet (folded from zero events) must not crash."""
    state = RunState(run_id=uuid4())
    result = BudgetEntryGate(GENEROUS).evaluate(state)
    assert result.verdict is GateVerdict.PASS


def test_exactly_at_the_limit_passes() -> None:
    """Boundary check: the limit itself is allowed, only exceeding it fails."""
    exact = BudgetLimits(max_tokens=100, max_usd=1.0, max_wallclock_seconds=3600)
    state = _state(input_tokens=100)
    assert BudgetEntryGate(exact).evaluate(state).verdict is GateVerdict.PASS
