"""Fake NodeExecutor and ApprovalProvider implementations for scheduler tests.

Not a test module itself (no `test_` prefix, not collected by pytest). These
stand in for the agent plane, which does not exist yet - this is exactly the
substitution Phase 1's exit criterion calls for: "the full graph executes end
to end with fake agents."
"""

from __future__ import annotations

import asyncio
import time

from ases.kernel.gates import ApprovalDecision
from ases.kernel.graph import NodeSpec
from ases.kernel.scheduler import NodeExecutionOutcome
from ases.kernel.state import RunState


class FixedExecutor:
    """Always returns the same outcome. Records how many times it was called."""

    def __init__(self, outcome: NodeExecutionOutcome) -> None:
        self.outcome = outcome
        self.call_count = 0

    async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome:
        self.call_count += 1
        return self.outcome


class SequenceExecutor:
    """Returns outcomes in order across successive calls; repeats the last one."""

    def __init__(self, outcomes: list[NodeExecutionOutcome]) -> None:
        self._outcomes = outcomes
        self.call_count = 0

    async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome:
        outcome = self._outcomes[min(self.call_count, len(self._outcomes) - 1)]
        self.call_count += 1
        return outcome


class SlowExecutor:
    """Sleeps before returning, recording exactly when it started and
    finished. Two `SlowExecutor`s dispatched concurrently will have
    *overlapping* intervals; dispatched sequentially, they cannot - this is
    what proves genuine concurrency without relying on a wall-clock threshold,
    which is inherently flaky under CPU contention or a loaded machine."""

    def __init__(self, outcome: NodeExecutionOutcome, delay: float) -> None:
        self.outcome = outcome
        self.delay = delay
        self.started_at: float | None = None
        self.finished_at: float | None = None

    async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome:
        self.started_at = time.monotonic()
        await asyncio.sleep(self.delay)
        self.finished_at = time.monotonic()
        return self.outcome

    def overlaps(self, other: SlowExecutor) -> bool:
        assert self.started_at is not None and self.finished_at is not None
        assert other.started_at is not None and other.finished_at is not None
        return self.started_at < other.finished_at and other.started_at < self.finished_at


class ScriptedApprovals:
    """Returns pre-scripted decisions per node id, one per call; None once exhausted."""

    def __init__(self, script: dict[str, list[ApprovalDecision | None]]) -> None:
        self._script = script
        self._index: dict[str, int] = {}

    async def decide(self, node: NodeSpec, state: RunState) -> ApprovalDecision | None:
        queue = self._script.get(node.id, [])
        i = self._index.get(node.id, 0)
        if i >= len(queue):
            return None
        self._index[node.id] = i + 1
        return queue[i]


def ok(kind: str = "Artifact", **payload: object) -> NodeExecutionOutcome:
    return NodeExecutionOutcome(ok=True, artifact_kind=kind, artifact_payload=payload or {"v": 1})


def fail(error: str = "boom") -> NodeExecutionOutcome:
    return NodeExecutionOutcome(ok=False, error=error)
