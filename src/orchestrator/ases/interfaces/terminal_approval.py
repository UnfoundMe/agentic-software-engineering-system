"""A minimal, real human-in-the-loop `kernel.gates.ApprovalProvider` for the
CLI (`ases run`): prints the pending artifact and prompts in this process's
own terminal, for every node that reaches `AWAITING_APPROVAL`.

**Why a blocking `input()` here is safe, not just convenient:**
`kernel.scheduler.Scheduler._step` calls and fully awaits `_request_approval`
(which is what calls `decide()` below) for every ready gate *before* it
gathers that step's batch of agent/tool nodes into `asyncio.gather` - so no
other node's `execute()` coroutine is ever in flight while this prompt is
blocking. `asyncio.to_thread` is used anyway, as the correct async idiom
rather than relying on that timing not to matter.
"""

from __future__ import annotations

import asyncio
import getpass

from ases.context.lineage import UnknownArtifactError
from ases.context.retriever import (
    ArtifactContentUnavailableError,
    ArtifactRef,
    ContextRetriever,
    UnknownArtifactKindError,
)
from ases.kernel.gates import ApprovalDecision
from ases.kernel.graph import NodeSpec
from ases.kernel.state import RunState


class TerminalApprovalProvider:
    """Implements `ApprovalProvider`. One instance is shared across every
    gate in a run; it holds no per-node state."""

    async def decide(self, node: NodeSpec, state: RunState) -> ApprovalDecision | None:
        record = state.approvals.get(node.id)
        artifact_hash = record.artifact_hash if record else ""

        print()
        print(f"=== Human approval required: {node.id} ===")
        if node.description:
            print(node.description)
        if artifact_hash:
            self._print_artifact(state, artifact_hash)
        else:
            print("(no artifact hash recorded for this approval)")

        answer = await asyncio.to_thread(input, "Approve? [y/N]: ")
        granted = answer.strip().lower() in ("y", "yes")
        reason = await asyncio.to_thread(input, "Reason (optional): ")
        return ApprovalDecision(granted=granted, actor=getpass.getuser(), reason=reason.strip())

    @staticmethod
    def _print_artifact(state: RunState, artifact_hash: str) -> None:
        retriever = ContextRetriever(state)
        try:
            artifact = retriever.fetch(ArtifactRef(artifact_hash=artifact_hash))
        except (
            UnknownArtifactError,
            UnknownArtifactKindError,
            ArtifactContentUnavailableError,
        ) as exc:
            print(f"(could not load artifact {artifact_hash}: {exc})")
            return
        print(f"--- {type(artifact).__name__} ({artifact_hash[:12]}...) ---")
        print(artifact.model_dump_json(indent=2))
