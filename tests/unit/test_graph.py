"""Graph validation.

Every rejection here happens at load or admission time. A graph that cannot
execute correctly must never start executing.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ases.kernel.graph import (
    Edge,
    EdgeCondition,
    GraphError,
    JoinPolicy,
    NodeKind,
    NodeSpec,
    RetryPolicy,
    WorkflowGraph,
)
from ases.kernel.state import NodeStatus


def agent(node_id: str, **kw: object) -> NodeSpec:
    return NodeSpec(id=node_id, kind=NodeKind.AGENT, handler=node_id, **kw)  # type: ignore[arg-type]


def terminal(node_id: str = "done") -> NodeSpec:
    return NodeSpec(id=node_id, kind=NodeKind.TERMINAL)


def linear_graph() -> WorkflowGraph:
    return WorkflowGraph(
        name="linear",
        entry=("a",),
        nodes=(agent("a"), agent("b"), terminal()),
        edges=(Edge(source="a", target="b"), Edge(source="b", target="done")),
    )


# --- happy path ---------------------------------------------------------


def test_a_valid_graph_validates() -> None:
    linear_graph().validate_graph()


def test_parallel_fan_out_and_join() -> None:
    graph = WorkflowGraph(
        name="parallel",
        entry=("split",),
        nodes=(
            agent("split"),
            agent("left"),
            agent("right"),
            NodeSpec(id="join", kind=NodeKind.BARRIER, join=JoinPolicy.ALL),
            terminal(),
        ),
        edges=(
            Edge(source="split", target="left"),
            Edge(source="split", target="right"),
            Edge(source="left", target="join"),
            Edge(source="right", target="join"),
            Edge(source="join", target="done"),
        ),
    )
    graph.validate_graph()
    assert len(graph.predecessors("join")) == 2


def test_multiple_entry_nodes_are_supported() -> None:
    """REQ_ANALYSIS and CODEBASE_INDEX start together; that is not an accident."""
    graph = WorkflowGraph(
        name="two-entries",
        entry=("req", "index"),
        nodes=(agent("req"), agent("index"), NodeSpec(id="j", kind=NodeKind.BARRIER), terminal()),
        edges=(
            Edge(source="req", target="j"),
            Edge(source="index", target="j"),
            Edge(source="j", target="done"),
        ),
    )
    graph.validate_graph()


# --- structural rejections ---------------------------------------------


def test_dangling_edge_is_rejected() -> None:
    graph = WorkflowGraph(
        name="bad",
        entry=("a",),
        nodes=(agent("a"), terminal()),
        edges=(Edge(source="a", target="nowhere"),),
    )
    with pytest.raises(GraphError, match="unknown target"):
        graph.validate_graph()


def test_duplicate_node_ids_are_rejected() -> None:
    graph = WorkflowGraph(
        name="bad",
        entry=("a",),
        nodes=(agent("a"), agent("a"), terminal()),
        edges=(Edge(source="a", target="done"),),
    )
    with pytest.raises(GraphError, match="duplicate node ids"):
        graph.validate_graph()


def test_entry_must_name_a_real_node() -> None:
    graph = WorkflowGraph(
        name="bad",
        entry=("ghost",),
        nodes=(agent("a"), terminal()),
        edges=(Edge(source="a", target="done"),),
    )
    with pytest.raises(GraphError, match="entry references unknown node"):
        graph.validate_graph()


def test_entry_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        WorkflowGraph(name="bad", entry=(), nodes=(terminal(),), edges=())


def test_unreachable_node_is_rejected() -> None:
    """An orphan means the author intended an edge that is not there."""
    graph = WorkflowGraph(
        name="bad",
        entry=("a",),
        nodes=(agent("a"), agent("orphan"), terminal()),
        edges=(Edge(source="a", target="done"), Edge(source="orphan", target="done")),
    )
    with pytest.raises(GraphError, match="unreachable"):
        graph.validate_graph()


def test_node_that_cannot_reach_a_terminal_is_rejected() -> None:
    """Otherwise a run can reach a state from which it can never finish."""
    graph = WorkflowGraph(
        name="bad",
        entry=("a",),
        nodes=(agent("a"), agent("dead_end"), terminal()),
        edges=(Edge(source="a", target="done"), Edge(source="a", target="dead_end")),
    )
    with pytest.raises(GraphError, match="no terminal is reachable"):
        graph.validate_graph()


def test_graph_without_a_terminal_is_rejected() -> None:
    graph = WorkflowGraph(name="bad", entry=("a",), nodes=(agent("a"),), edges=())
    with pytest.raises(GraphError, match="no terminal node"):
        graph.validate_graph()


# --- cycles -------------------------------------------------------------


def _repair_loop(*, budget: int | None) -> WorkflowGraph:
    return WorkflowGraph(
        name="repair-loop",
        entry=("test",),
        nodes=(agent("test", cycle_budget=budget), agent("repair"), terminal()),
        edges=(
            Edge(source="test", target="repair", condition=EdgeCondition.ON_FAILURE),
            Edge(source="repair", target="test", condition=EdgeCondition.ALWAYS),
            Edge(source="test", target="done"),
        ),
    )


def test_unbounded_cycle_is_rejected() -> None:
    """A repair loop with no budget is a run that cannot be shown to terminate."""
    with pytest.raises(GraphError, match="cycle_budget"):
        _repair_loop(budget=None).validate_graph()


def test_bounded_cycle_is_accepted() -> None:
    """Cycles are legitimate - the repair loop is one. Unbounded ones are not."""
    _repair_loop(budget=3).validate_graph()


def test_self_loop_without_budget_is_rejected() -> None:
    graph = WorkflowGraph(
        name="bad",
        entry=("a",),
        nodes=(agent("a"), terminal()),
        edges=(
            Edge(source="a", target="a", condition=EdgeCondition.ON_FAILURE),
            Edge(source="a", target="done"),
        ),
    )
    with pytest.raises(GraphError, match="cycle_budget"):
        graph.validate_graph()


# --- node specs ---------------------------------------------------------


def test_quorum_requires_a_value() -> None:
    with pytest.raises(ValidationError, match="requires `quorum`"):
        NodeSpec(id="j", kind=NodeKind.BARRIER, join=JoinPolicy.QUORUM)


def test_quorum_cannot_exceed_incoming_edges() -> None:
    """Otherwise the node can never be satisfied and the run stalls silently."""
    graph = WorkflowGraph(
        name="bad",
        entry=("a",),
        nodes=(
            agent("a"),
            NodeSpec(id="j", kind=NodeKind.BARRIER, join=JoinPolicy.QUORUM, quorum=3),
            terminal(),
        ),
        edges=(Edge(source="a", target="j"), Edge(source="j", target="done")),
    )
    with pytest.raises(GraphError, match="quorum 3 exceeds"):
        graph.validate_graph()


def test_agent_node_requires_a_handler() -> None:
    with pytest.raises(ValidationError, match="requires `handler`"):
        NodeSpec(id="a", kind=NodeKind.AGENT)


def test_node_id_rejects_unsafe_characters() -> None:
    """Node ids reach file paths and log fields; keep the alphabet narrow."""
    with pytest.raises(ValidationError):
        NodeSpec(id="../escape", kind=NodeKind.TERMINAL)


# --- edge conditions ----------------------------------------------------


@pytest.mark.parametrize(
    ("condition", "status", "expected"),
    [
        (EdgeCondition.ON_SUCCESS, NodeStatus.SUCCEEDED, True),
        (EdgeCondition.ON_SUCCESS, NodeStatus.FAILED, False),
        (EdgeCondition.ON_FAILURE, NodeStatus.FAILED, True),
        (EdgeCondition.ON_FAILURE, NodeStatus.FALLBACK, True),
        (EdgeCondition.ON_REJECTED, NodeStatus.REJECTED, True),
        (EdgeCondition.ON_STALE, NodeStatus.STALE, True),
        (EdgeCondition.ALWAYS, NodeStatus.CANCELLED, True),
    ],
)
def test_edge_conditions(condition: EdgeCondition, status: NodeStatus, expected: bool) -> None:
    assert condition.matches(status) is expected


# --- retry policy -------------------------------------------------------


def test_first_attempt_never_waits() -> None:
    assert RetryPolicy().delay_for(1) == 0.0


def test_backoff_grows_and_is_capped() -> None:
    policy = RetryPolicy(backoff_seconds=1.0, backoff_multiplier=2.0, max_backoff_seconds=5.0)
    assert [policy.delay_for(n) for n in (2, 3, 4, 5, 6)] == [1.0, 2.0, 4.0, 5.0, 5.0]


# --- dynamic subgraph admission ----------------------------------------


def _decompose_graph() -> WorkflowGraph:
    return WorkflowGraph(
        name="g",
        entry=("decompose",),
        nodes=(agent("decompose"), NodeSpec(id="barrier", kind=NodeKind.BARRIER), terminal()),
        edges=(Edge(source="decompose", target="barrier"), Edge(source="barrier", target="done")),
    )


def test_valid_subgraph_is_admitted() -> None:
    expanded = _decompose_graph().with_subgraph(
        nodes=[agent("task1"), agent("task2")],
        edges=[
            Edge(source="decompose", target="task1"),
            Edge(source="decompose", target="task2"),
            Edge(source="task1", target="barrier"),
            Edge(source="task2", target="barrier"),
        ],
    )
    expanded.validate_graph()
    assert {n.id for n in expanded.nodes} >= {"task1", "task2"}


def test_subgraph_that_would_create_an_unbounded_cycle_is_refused() -> None:
    """Admission is where a decomposer's proposal is checked - before it runs."""
    with pytest.raises(GraphError, match="cycle_budget"):
        linear_graph().with_subgraph(
            nodes=[agent("loop")],
            edges=[
                Edge(source="a", target="loop"),
                Edge(source="loop", target="a", condition=EdgeCondition.ALWAYS),
            ],
        )


def test_subgraph_with_a_dangling_edge_is_refused() -> None:
    with pytest.raises(GraphError, match="unknown target"):
        _decompose_graph().with_subgraph(
            nodes=[agent("task1")],
            edges=[Edge(source="decompose", target="task1"), Edge(source="task1", target="ghost")],
        )


def test_subgraph_cannot_redefine_an_existing_node() -> None:
    """Silently replacing a node would rewrite approved history."""
    with pytest.raises(GraphError, match="redefines existing node"):
        linear_graph().with_subgraph(nodes=[agent("a")], edges=[])


def test_admission_leaves_the_original_graph_untouched() -> None:
    graph = _decompose_graph()
    before = len(graph.nodes)
    graph.with_subgraph(
        nodes=[agent("t1")],
        edges=[Edge(source="decompose", target="t1"), Edge(source="t1", target="barrier")],
    )
    assert len(graph.nodes) == before
