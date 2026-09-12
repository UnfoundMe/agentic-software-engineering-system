"""Fake NodeExecutor and ApprovalProvider implementations for scheduler tests.

Not a test module itself (no `test_` prefix, not collected by pytest). These
stand in for the agent plane, which does not exist yet - this is exactly the
substitution Phase 1's exit criterion calls for: "the full graph executes end
to end with fake agents."
"""

from __future__ import annotations

import asyncio

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
    """Sleeps before returning - used to prove genuine concurrent dispatch."""

    def __init__(self, outcome: NodeExecutionOutcome, delay: float) -> None:
        self.outcome = outcome
        self.delay = delay

    async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome:
        await asyncio.sleep(self.delay)
        return self.outcome


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
