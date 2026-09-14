"""The workflow graph.

The graph is the orchestrator's control flow, and it is **data**, not code: a
YAML document validated on load. Agents never modify it. The only runtime change
permitted is subgraph admission (`with_subgraph`), where a decomposer proposes
nodes that are schema-checked, cycle-checked and policy-checked *before* they
can execute - bounded autonomy rather than unbounded.

Edge conditions are a closed enum rather than expressions. An expression
language here would be an injection surface reachable from model output and
would make control flow non-deterministic; neither is acceptable in the one
component that must stay predictable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ases.kernel.state import TERMINAL_STATUSES, NodeStatus


class GraphError(ValueError):
    """The graph is not executable. Raised at load time, never at run time."""


class NodeKind(StrEnum):
    AGENT = "agent"  # dispatches to an LLM-backed worker
    TOOL = "tool"  # dispatches to a registered tool, no model involved
    GATE = "gate"  # pure decision point; may require human approval
    BARRIER = "barrier"  # synchronisation only
    TERMINAL = "terminal"


class JoinPolicy(StrEnum):
    ALL = "all"
    ANY = "any"
    QUORUM = "quorum"


class EdgeCondition(StrEnum):
    """When an edge is traversable, evaluated against the source node's state."""

    ON_SUCCESS = "on_success"
    ON_FAILURE = "on_failure"
    ON_REJECTED = "on_rejected"  # a human rejected the gate - clarification cycle
    ON_STALE = "on_stale"
    ALWAYS = "always"

    def matches(self, status: NodeStatus) -> bool:
        match self:
            case EdgeCondition.ALWAYS:
                # "Always" means "regardless of which outcome the source
                # reached" - the union of every other condition's match set -
                # never "regardless of whether the source has run at all".
                # A literal `True` here was a real bug, found via
                # `test_greenfield_full_e2e.py`: combined with `join: any`,
                # it made a node with an `always`-conditioned incoming edge
                # from a *not-yet-run* predecessor (still at its default
                # `PENDING`) immediately "ready", regardless of its other,
                # real predecessor's progress - `test_run`'s dormant
                # `repair -[always]-> test_run` edge fired before `test_gen`
                # (its actual, intended predecessor) ever ran. Masked in the
                # one existing test of this edge shape
                # (`test_scheduler.py::_repair_graph`) because there the
                # affected node is the graph's *entry point*, which bypasses
                # join checking entirely on its first pass.
                #
                # The match set is every other condition's set, plus
                # `TERMINAL_STATUSES` (a cancelled/skipped/halted node also
                # reached a definitive outcome, just not one any `ON_*`
                # condition names) - deliberately still excluding
                # `PENDING`/`READY`/`RUNNING`/every other in-flight or
                # not-yet-started status.
                return status in TERMINAL_STATUSES or status in (
                    NodeStatus.SUCCEEDED,
                    NodeStatus.FAILED,
                    NodeStatus.ROLLED_BACK,
                    NodeStatus.FALLBACK,
                    NodeStatus.REJECTED,
                    NodeStatus.STALE,
                )
            case EdgeCondition.ON_SUCCESS:
                return status is NodeStatus.SUCCEEDED
            case EdgeCondition.ON_FAILURE:
                return status in (NodeStatus.FAILED, NodeStatus.ROLLED_BACK, NodeStatus.FALLBACK)
            case EdgeCondition.ON_REJECTED:
                return status is NodeStatus.REJECTED
            case EdgeCondition.ON_STALE:
                return status is NodeStatus.STALE
        raise AssertionError(f"unhandled condition {self}")  # pragma: no cover


class Exhausted(StrEnum):
    """What to do when a node's retry or cycle budget runs out."""

    SAFE_STOP = "safe_stop"
    FALLBACK = "fallback"
    FAIL_RUN = "fail_run"


class RetryPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(default=1, ge=1, le=10)
    backoff_seconds: float = Field(default=1.0, ge=0.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    max_backoff_seconds: float = Field(default=60.0, ge=0.0)

    def delay_for(self, attempt: int) -> float:
        """Delay before `attempt` (1-based). Attempt 1 never waits."""
        if attempt <= 1:
            return 0.0
        raw = self.backoff_seconds * (self.backoff_multiplier ** (attempt - 2))
        return min(raw, self.max_backoff_seconds)


class NodeSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")
    kind: NodeKind
    #: Agent or tool name. Resolved by the runner, never by the kernel - the
    #: kernel must not know what agents exist.
    handler: str | None = None
    join: JoinPolicy = JoinPolicy.ALL
    quorum: int | None = None
    entry_gates: tuple[str, ...] = ()
    exit_gates: tuple[str, ...] = ()
    produces: str | None = None  # artifact kind, for lineage
    retry: RetryPolicy = RetryPolicy()
    #: How long the workflow author expects this node to need. Advisory: the
    #: scheduler does **not** cancel a node that exceeds it, and deliberately
    #: so. Execution is already bounded at the two layers that can attribute a
    #: stall to something specific - `kernel.tools.registry` wraps every tool
    #: invocation in `asyncio.wait_for(..., spec.timeout_s)`, and the provider
    #: SDK bounds each HTTP call - and a third ceiling on top of those could
    #: only cut short work those layers considered healthy. It briefly did
    #: exactly that: a live run cancelled `scaffold` at its declared 120s on a
    #: call that had taken 201s successfully the run before, ending the run
    #: after two human approvals. Read this as documentation of intent, not as
    #: a guarantee; `kernel/scheduler.py`'s module docstring states the
    #: residual gap this leaves.
    timeout_seconds: float = Field(default=300.0, gt=0)
    #: Required on at least one node of every cycle. Without it a repair loop
    #: could run forever, so an unbudgeted cycle is rejected at load time.
    cycle_budget: int | None = Field(default=None, ge=1)
    on_exhausted: Exhausted = Exhausted.SAFE_STOP
    #: Human approval is required to leave this node. The approval binds to the
    #: hash of the artifact the node produced.
    requires_approval: bool = False
    description: str = ""

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.join is JoinPolicy.QUORUM and self.quorum is None:
            raise ValueError(f"node {self.id!r}: join=quorum requires `quorum`")
        if self.join is not JoinPolicy.QUORUM and self.quorum is not None:
            raise ValueError(f"node {self.id!r}: `quorum` is only meaningful with join=quorum")
        if self.kind in (NodeKind.AGENT, NodeKind.TOOL) and not self.handler:
            raise ValueError(f"node {self.id!r}: kind={self.kind} requires `handler`")
        return self


class Edge(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    target: str
    condition: EdgeCondition = EdgeCondition.ON_SUCCESS

    def __str__(self) -> str:
        return f"{self.source} -[{self.condition}]-> {self.target}"


class WorkflowGraph(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    nodes: tuple[NodeSpec, ...]
    edges: tuple[Edge, ...]
    #: Where a run begins. Declared, not inferred from in-degree.
    #:
    #: Inference breaks precisely where it matters: once a node participates in
    #: a cycle - which every repair loop and clarification cycle creates - it
    #: has an incoming edge and would stop looking like a start node. Declaring
    #: entries also lets a workflow have several (REQ_ANALYSIS alongside
    #: CODEBASE_INDEX) without that being an accident of edge layout.
    entry: tuple[str, ...] = Field(min_length=1)

    # -- construction --------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise GraphError(f"{path}: expected a mapping at the top level")
        graph = cls.model_validate(raw)
        graph.validate_graph()
        return graph

    # -- indexes -------------------------------------------------------

    @property
    def by_id(self) -> Mapping[str, NodeSpec]:
        return {n.id: n for n in self.nodes}

    def node(self, node_id: str) -> NodeSpec:
        try:
            return self.by_id[node_id]
        except KeyError:
            raise GraphError(f"unknown node {node_id!r}") from None

    def successors(self, node_id: str) -> tuple[Edge, ...]:
        return tuple(e for e in self.edges if e.source == node_id)

    def predecessors(self, node_id: str) -> tuple[Edge, ...]:
        return tuple(e for e in self.edges if e.target == node_id)

    @property
    def entry_nodes(self) -> tuple[str, ...]:
        return self.entry

    @property
    def terminal_nodes(self) -> tuple[str, ...]:
        return tuple(n.id for n in self.nodes if n.kind is NodeKind.TERMINAL)

    # -- validation ----------------------------------------------------

    def validate_graph(self) -> None:
        """Reject a graph that could not execute correctly.

        Runs at load time and at subgraph admission, never mid-node. An
        orchestrator that discovers its graph is unexecutable halfway through a
        run has already done damage it cannot explain.
        """
        ids = [n.id for n in self.nodes]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise GraphError(f"duplicate node ids: {sorted(duplicates)}")
        known = set(ids)
        if not known:
            raise GraphError("graph has no nodes")

        for edge in self.edges:
            if edge.source not in known:
                raise GraphError(f"edge {edge} has unknown source")
            if edge.target not in known:
                raise GraphError(f"edge {edge} has unknown target")

        unknown_entries = set(self.entry) - known
        if unknown_entries:
            raise GraphError(f"entry references unknown node(s): {sorted(unknown_entries)}")
        entries = self.entry

        terminals = self.terminal_nodes
        if not terminals:
            raise GraphError("graph has no terminal node")

        unreachable = known - self._reachable_from(entries)
        if unreachable:
            raise GraphError(f"nodes unreachable from an entry node: {sorted(unreachable)}")

        dead_ends = known - self._can_reach(terminals)
        if dead_ends:
            raise GraphError(f"nodes from which no terminal is reachable: {sorted(dead_ends)}")

        self._validate_cycle_budgets()

        for node in self.nodes:
            if node.join is JoinPolicy.QUORUM:
                incoming = len(self.predecessors(node.id))
                assert node.quorum is not None  # guaranteed by NodeSpec validator
                if node.quorum > incoming:
                    raise GraphError(
                        f"node {node.id!r}: quorum {node.quorum} exceeds "
                        f"{incoming} incoming edge(s)"
                    )

    def _reachable_from(self, starts: Iterable[str]) -> set[str]:
        seen: set[str] = set()
        stack = list(starts)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(e.target for e in self.successors(current))
        return seen

    def _can_reach(self, targets: Iterable[str]) -> set[str]:
        seen: set[str] = set()
        stack = list(targets)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(e.source for e in self.predecessors(current))
        return seen

    def _validate_cycle_budgets(self) -> None:
        """Every cycle must be bounded by at least one node's `cycle_budget`.

        Cycles are legitimate and necessary here - the repair loop and the
        clarification cycle are both backward edges. An *unbounded* cycle is
        not: it is a run that can never be shown to terminate.
        """
        by_id = self.by_id
        for component in self._strongly_connected_components():
            is_cycle = len(component) > 1 or any(
                e.source == e.target for e in self.edges if e.source in component
            )
            if not is_cycle:
                continue
            if not any(by_id[nid].cycle_budget is not None for nid in component):
                raise GraphError(
                    f"cycle {sorted(component)} has no cycle_budget on any member; "
                    "an unbounded cycle cannot be shown to terminate"
                )

    def _strongly_connected_components(self) -> list[set[str]]:
        """Tarjan's algorithm, iterative to avoid recursion limits on wide graphs."""
        index_of: dict[str, int] = {}
        low: dict[str, int] = {}
        on_stack: set[str] = set()
        stack: list[str] = []
        components: list[set[str]] = []
        counter = 0

        adjacency: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for edge in self.edges:
            adjacency[edge.source].append(edge.target)

        for root in adjacency:
            if root in index_of:
                continue
            work: list[tuple[str, int]] = [(root, 0)]
            while work:
                node, child_index = work[-1]
                if child_index == 0:
                    index_of[node] = low[node] = counter
                    counter += 1
                    stack.append(node)
                    on_stack.add(node)

                if child_index < len(adjacency[node]):
                    work[-1] = (node, child_index + 1)
                    child = adjacency[node][child_index]
                    if child not in index_of:
                        work.append((child, 0))
                    elif child in on_stack:
                        low[node] = min(low[node], index_of[child])
                else:
                    work.pop()
                    if work:
                        parent = work[-1][0]
                        low[parent] = min(low[parent], low[node])
                    if low[node] == index_of[node]:
                        component: set[str] = set()
                        while True:
                            member = stack.pop()
                            on_stack.discard(member)
                            component.add(member)
                            if member == node:
                                break
                        components.append(component)
        return components

    # -- dynamic subgraph admission ------------------------------------

    def with_subgraph(self, nodes: Sequence[NodeSpec], edges: Sequence[Edge]) -> WorkflowGraph:
        """Return a new graph including a proposed subgraph.

        Validated before it is returned, so a decomposer's proposal that would
        create an unreachable node, a dangling edge or an unbounded cycle is
        rejected at admission - it never executes.
        """
        existing = self.by_id
        for node in nodes:
            if node.id in existing:
                raise GraphError(f"subgraph redefines existing node {node.id!r}")
        candidate = WorkflowGraph(
            name=self.name,
            description=self.description,
            nodes=(*self.nodes, *nodes),
            edges=(*self.edges, *edges),
            entry=self.entry,
        )
        candidate.validate_graph()
        return candidate

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
