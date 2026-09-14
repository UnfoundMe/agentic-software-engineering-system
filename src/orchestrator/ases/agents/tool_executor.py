"""Adapts a registered Phase 3 tool into the scheduler's `NodeExecutor`
protocol, for `kind: tool` graph nodes (docs/03 section 3.3: `test_run`,
`sec_scan`) - the direct counterpart to `AgentNodeExecutor` for a node with
no LLM involved at all.

Lives here, in `agents/`, for the same reason `executor.py` does: it imports
`kernel.tools.registry` and `context.retriever`, and `kernel/` must never
import back into either.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import PurePosixPath

from ases.context.retriever import ContextRetriever
from ases.kernel.failures import FailureKind
from ases.kernel.graph import NodeSpec
from ases.kernel.scheduler import NodeExecutionOutcome
from ases.kernel.state import RunState
from ases.kernel.tools.classification import ToolContext
from ases.kernel.tools.registry import ToolRegistry

#: Builds the tool's `args` mapping from the current run's folded state - the
#: tool-node counterpart to `agents.base.Agent.build_input`. Takes a
#: `ContextRetriever` (never the raw `RunState`) for the same scoping reason
#: every agent does, plus the `NodeSpec` being executed.
#:
#: The node argument is what lets **one** registered handler serve many
#: nodes. Before dynamic task admission there were three `dotnet_build_*`
#: handlers, one per hard-coded build node, each closing over its own
#: project - which cannot work when the build nodes are admitted at runtime
#: and there are as many of them as the decomposer proposed. The builder now
#: resolves its arguments from the node it is actually running for.
ArgsBuilder = Callable[[ContextRetriever, NodeSpec], Mapping[str, object]]

#: Builds an artifact payload (or `None`, meaning "no artifact this time")
#: from a successful tool invocation's output - e.g. turning
#: `security.scan_for_secrets`'s `{"findings": [...]}` into a `PolicyViolation`
#: only when there is something to report.
ArtifactBuilder = Callable[[Mapping[str, object]], Mapping[str, object] | None]


def _no_args(retriever: ContextRetriever, node: NodeSpec) -> Mapping[str, object]:
    return {}


def _no_artifact(output: Mapping[str, object]) -> Mapping[str, object] | None:
    return None


class ToolNodeExecutor:
    """One instance per `kind: tool` graph node. `artifact_kind`/`build_artifact`
    are both optional - a tool node with nothing to report back as a tracked
    artifact (e.g. `test_run`, which `workflows/greenfield.yaml` gives no
    `produces:` at all) simply reports `ok`/`error`."""

    def __init__(
        self,
        tool_name: str,
        *,
        tools: ToolRegistry,
        tool_cwd: PurePosixPath,
        build_args: ArgsBuilder = _no_args,
        artifact_kind: str | None = None,
        build_artifact: ArtifactBuilder = _no_artifact,
        failure_kind: FailureKind = FailureKind.TOOL_FAILURE,
    ) -> None:
        self._tool_name = tool_name
        self._tools = tools
        self._tool_cwd = tool_cwd
        self._build_args = build_args
        self._artifact_kind = artifact_kind
        self._build_artifact = build_artifact
        #: How a failure of *this* tool should be classified. Supplied by the
        #: wiring, which is the layer that knows a given node invokes
        #: `dotnet.build` rather than, say, a security scan - never inferred
        #: here from the tool's name, and never decided by the kernel.
        #: `BUILD_FAILURE` is the expected, designed-for outcome the repair
        #: cycle exists to act on, and reads very differently in an event log
        #: from a tool that broke.
        self._failure_kind = failure_kind

    async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome:
        retriever = ContextRetriever(state)
        args = self._build_args(retriever, node)
        tool_ctx = ToolContext(cwd=self._tool_cwd, run_id=str(state.run_id), node_id=node.id)
        result = await self._tools.invoke(self._tool_name, args, tool_ctx)

        if result.denied:
            return NodeExecutionOutcome(
                ok=False,
                failure_kind=FailureKind.POLICY_DENIED,
                error=f"tool {self._tool_name!r} denied: {result.error}",
            )

        artifact_payload = None
        if self._artifact_kind is not None:
            # Deliberately consulted regardless of `result.ok`: a security
            # scan's most important artifact is exactly the one produced
            # when it *fails* (findings exist) - `build_artifact` decides
            # from the tool's own output, not from success/failure.
            artifact_payload = self._build_artifact(result.output)

        return NodeExecutionOutcome(
            ok=result.ok,
            failure_kind=self._failure_kind,
            error=result.error,
            artifact_kind=self._artifact_kind if artifact_payload is not None else None,
            artifact_payload=artifact_payload,
        )
