"""Entry/exit gates and the human approval boundary.

Two gate mechanisms exist and they are deliberately different shapes:

- The **entry gate** implemented here (`BudgetEntryGate`) is deterministic and
  needs no agent: it checks the run's accumulated token/cost/wall-clock spend
  against configured limits. This is real enforcement, not a placeholder - a
  run that has burned its budget is refused further dispatch and safe-stops.

- **Human approval** is not a `Gate` at all; it is modelled as an
  `ApprovalProvider` the scheduler calls when a node declares
  `requires_approval=True`. A provider may answer immediately (a fake, for
  tests) or return `None` to mean "no decision yet" (a real dashboard, where a
  human has not acted) - in which case the scheduler leaves the node
  `AWAITING_APPROVAL` and moves on; a later call to `Scheduler.run` picks it
  back up once a decision exists.

Policy-based gates (secret scanning, change-control rules) belong to
`kernel.policy` (Phase 3) and are not implemented here.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ases.kernel.events import utc_now
from ases.kernel.graph import NodeSpec
from ases.kernel.state import RunState


class GateVerdict(StrEnum):
    PASS = "pass"  # noqa: S105 - a verdict, not a credential
    FAIL = "fail"
    ESCALATE = "escalate"


class GateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    verdict: GateVerdict
    reason: str = ""


class BudgetLimits(BaseModel):
    """Mirrors the relevant `Settings` fields without importing `config`.

    Kernel modules take plain values rather than the `Settings` object itself,
    so kernel tests never need a `.env` or a working directory layout.
    """

    model_config = ConfigDict(frozen=True)

    max_tokens: int
    max_usd: float
    max_wallclock_seconds: float


class BudgetEntryGate:
    """Refuses dispatch once the run has exceeded its configured budget.

    Applied uniformly to every node, not just LLM-backed ones: wall-clock spend
    accrues regardless of which node is consuming it.
    """

    def __init__(self, limits: BudgetLimits) -> None:
        self._limits = limits

    def evaluate(self, state: RunState, *, now: datetime | None = None) -> GateResult:
        if state.usage.total_tokens > self._limits.max_tokens:
            return GateResult(
                verdict=GateVerdict.FAIL,
                reason=(
                    f"token budget exceeded: {state.usage.total_tokens} > {self._limits.max_tokens}"
                ),
            )
        if state.usage.usd > self._limits.max_usd:
            return GateResult(
                verdict=GateVerdict.FAIL,
                reason=(
                    f"cost budget exceeded: ${state.usage.usd:.2f} > ${self._limits.max_usd:.2f}"
                ),
            )
        if state.created_at is not None:
            elapsed = ((now or utc_now()) - state.created_at).total_seconds()
            if elapsed > self._limits.max_wallclock_seconds:
                return GateResult(
                    verdict=GateVerdict.FAIL,
                    reason=(
                        f"wall-clock budget exceeded: {elapsed:.0f}s > "
                        f"{self._limits.max_wallclock_seconds:.0f}s"
                    ),
                )
        return GateResult(verdict=GateVerdict.PASS)


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    granted: bool
    actor: str
    reason: str = ""


@runtime_checkable
class ApprovalProvider(Protocol):
    """Supplies human decisions. The scheduler never assumes one is available.

    Returning `None` means "not decided yet" - a legitimate, expected outcome
    for a real human-in-the-loop run, not an error.
    """

    async def decide(self, node: NodeSpec, state: RunState) -> ApprovalDecision | None: ...
