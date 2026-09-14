"""The checked-in workflow YAML files load and validate."""

from __future__ import annotations

from pathlib import Path

import ases
from ases.kernel.graph import EdgeCondition, JoinPolicy, WorkflowGraph

WORKFLOWS_DIR = Path(ases.__file__).parent / "workflows"


def test_greenfield_workflow_loads_and_validates() -> None:
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    assert graph.name == "greenfield"
    assert graph.entry == ("req",)
    assert "summary" in graph.terminal_nodes


def test_greenfield_workflow_has_the_documented_gates() -> None:
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    gate_ids = {n.id for n in graph.nodes if n.kind.value == "gate" and n.requires_approval}
    assert gate_ids == {"gate1", "gate2", "gate3", "migration_gate"}


def test_greenfield_workflow_has_a_bounded_repair_cycle() -> None:
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    test_run = graph.node("test_run")
    assert test_run.cycle_budget is not None


def test_greenfield_workflow_declares_no_implementation_architecture() -> None:
    """The static workflow describes the engineering *lifecycle* and nothing
    about the system being built. It used to hard-code `impl_domain`,
    `impl_api` and `impl_infrastructure` with their own build and repair
    nodes - one particular architecture frozen into the generic greenfield
    path, which is why live run `009ea59f-...` silently never implemented the
    Application layer its own decomposer had planned for.

    Implementation nodes are admitted at runtime from the decomposer's
    TaskGraph (`agents/planner.py`). Nothing named after a layer may reappear
    in this file."""
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    declared = {n.id for n in graph.nodes}
    architecture_shaped = {
        node_id
        for node_id in declared
        for layer in ("domain", "application", "infrastructure", "api", "web", "persistence")
        if layer in node_id.lower()
    }
    assert architecture_shaped == set(), (
        f"{sorted(architecture_shaped)} name parts of one system's architecture. The "
        "implementation subgraph belongs to the decomposer, not to this file."
    )
    assert {"impl_start", "impl_end"} <= declared
    assert graph.node("impl_end").join is JoinPolicy.ALL
    assert any(e.source == "impl_start" and e.target == "impl_end" for e in graph.edges), (
        "impl_start -> impl_end keeps the file a valid graph with an empty TaskGraph"
    )


def test_the_execution_stage_sits_between_decompose_and_the_rest_of_the_lifecycle() -> None:
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    assert {e.source for e in graph.predecessors("impl_start")} == {"structure_check"}
    assert {e.source for e in graph.predecessors("structure_check")} == {"decompose"}
    assert {e.target for e in graph.successors("impl_end")} == {"migration", "barrier"}


def test_greenfield_workflow_gates_migration_apply_behind_human_approval() -> None:
    """docs/04 section 4.1: the agent generates a migration, it does not
    apply one - `ef.database_update` runs only after `migration_gate`."""
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    predecessors = {e.source for e in graph.predecessors("migration_apply")}
    assert predecessors == {"migration_gate"}
    assert graph.node("migration_gate").requires_approval is True


def test_greenfield_workflow_has_a_bounded_clarification_cycle() -> None:
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    req = graph.node("req")
    assert req.cycle_budget is not None


def test_every_gate_has_a_bounded_rejection_cycle_back_to_its_producer() -> None:
    """Found via a live run: only `gate1` used to have an `on_rejected` edge,
    so rejecting `gate2`/`migration_gate`/`gate3` silently stalled the run
    with no way to try again (docs/02 Phase 5/7's absence, concretely). Every
    gate now loops back to the node whose output it approves, each bounded."""
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    expected_producer_by_gate = {
        "gate1": "req",
        "gate2": "arch",
        "migration_gate": "migration",
        "gate3": "release",
    }
    for gate_id, producer_id in expected_producer_by_gate.items():
        on_rejected_targets = {
            e.target for e in graph.successors(gate_id) if e.condition is EdgeCondition.ON_REJECTED
        }
        assert on_rejected_targets == {producer_id}, gate_id
        assert graph.node(producer_id).cycle_budget is not None, producer_id


def test_every_redo_producer_joins_on_any_not_all() -> None:
    """A node with two incoming edges - the normal forward edge and a
    backward `on_rejected` edge - must use `join: any`: `join: all` (the
    default) would require both edges to match simultaneously, which is
    impossible on the very first pass since the `on_rejected` edge's source
    has not even run yet. This exact bug was caught by
    `test_greenfield_full_e2e.py` going from COMPLETED to FAILED the moment
    these edges were added without also setting `join: any`.

    `migration` is deliberately excluded here - see the next test."""
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    for node_id in ("arch", "release"):
        assert graph.node(node_id).join is JoinPolicy.ANY, node_id


def test_migration_waits_for_the_whole_execution_stage_or_a_rejection() -> None:
    """`migration` used to join `quorum: 2` over `build_domain` and
    `build_infrastructure` - two nodes named after one architecture's layers,
    chosen because docs/04 section 4.2 step 1 needs entities *and* a
    `DbContext` before `dotnet ef migrations add` can do anything.

    With the implementation subgraph admitted at runtime neither node exists,
    and no fixed pair could be named anyway. `impl_end` is the correct
    prerequisite and a stronger one: it joins `all`, so it is reached only
    once *every* admitted build has succeeded - entities and DbContext
    included, whatever projects this architecture put them in. That leaves
    one real prerequisite plus the usual `on_rejected` redo edge, which is
    exactly the `join: any` shape `arch` and `release` already use."""
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    migration = graph.node("migration")
    assert migration.join is JoinPolicy.ANY
    # One real prerequisite; the other two incoming edges are both backward -
    # the gate's rejection redo and its own failure retry.
    assert {e.source for e in graph.predecessors("migration")} == {
        "impl_end",
        "migration_gate",
        "migration",
    }
    assert graph.node("impl_end").join is JoinPolicy.ALL


def test_every_post_approval_producer_can_recover_from_its_own_failure() -> None:
    """`migration`, `test_gen`, `review`, `docs_gen` and `release` all run
    after gate1 and gate2 have been granted and after every implementation
    task has been written and compiled. Until these edges existed a single
    agent-protocol failure at any of them - a response truncated at
    `max_tokens`, which is what live runs actually produced - emitted
    RUN_FAILED and threw away two human approvals plus the entire build.

    Each now loops back to itself on failure, bounded by its own
    `cycle_budget`, so the run spends a retry instead of the approvals. The
    end-to-end behaviour is pinned in `test_greenfield_full_e2e.py`; this
    pins the shape in the file itself."""
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    for node_id in ("scaffold", "migration", "test_gen", "review", "docs_gen", "release"):
        failure_targets = {
            e.target
            for e in graph.successors(node_id)
            if e.condition in (EdgeCondition.ON_FAILURE, EdgeCondition.ALWAYS)
        }
        assert failure_targets == {node_id}, node_id
        assert graph.node(node_id).cycle_budget is not None, node_id


def test_a_self_failure_edge_forces_join_any_without_weakening_a_real_join() -> None:
    """The mechanical constraint behind the test above, and the way to get it
    wrong. A self-edge is an *incoming* edge, so under the default
    `join: all` the node's own `on_failure` edge would have to match its own
    status before it could ever start - and on first arrival that status is
    PENDING, which `on_failure` does not match. The node would never run at
    all.

    `join: any` is therefore required, and is only safe because each of these
    five has exactly one *forward* predecessor: `any` and `all` agree on the
    first entry. The two nodes that do join several real predecessors -
    `barrier` and `quality_gate` - are untouched and must stay `all`, or a
    failed producer would stop blocking them."""
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    backward = {
        ("migration", "migration_gate"),  # on_rejected redo
        ("release", "gate3"),  # on_rejected redo
    }
    for node_id in ("scaffold", "migration", "test_gen", "review", "docs_gen", "release"):
        assert graph.node(node_id).join is JoinPolicy.ANY, node_id
        forward = {
            e.source
            for e in graph.predecessors(node_id)
            if e.source != node_id and (node_id, e.source) not in backward
        }
        assert len(forward) == 1, (node_id, forward)

    for fan_in in ("barrier", "quality_gate"):
        assert graph.node(fan_in).join is JoinPolicy.ALL, fan_in
        assert len(graph.predecessors(fan_in)) > 1, fan_in
    # `impl_end` joins ALL too, but over edges admitted at runtime - it has
    # only `impl_start` statically. Asserted in its own test above.
    assert graph.node("impl_end").join is JoinPolicy.ALL


def test_the_deterministic_security_scan_is_deliberately_not_retried() -> None:
    """`sec_scan` is a tool node, not an agent: a deterministic scan that
    failed will fail again the same way, so a retry would only spend time to
    reach the same halt. Stated as a test so the omission reads as a decision
    rather than as the same oversight the five agent nodes had."""
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    assert graph.node("sec_scan").kind.value == "tool"
    assert not any(
        e.condition in (EdgeCondition.ON_FAILURE, EdgeCondition.ALWAYS)
        for e in graph.successors("sec_scan")
    )


def test_every_agent_downstream_of_an_approval_can_recover_from_its_own_failure() -> None:
    """The rule behind the hand-written list in the test above, computed from
    the graph instead of trusted. Any *agent* reachable only after an approval
    gate has been granted must have a recovery edge: without one, a single
    model slip throws away human decisions that cannot be re-derived.

    `scaffold` is why this is computed. It reads as early-lifecycle work and
    was filed with `req`/`arch` as "before any approval is spent", but its
    only predecessor is `gate2`. Live run `bd184488-...` failed there with
    both gates granted and ended on the spot.

    Two deliberate exclusions, both stated here so they read as decisions:

    - **Tool nodes** (`structure_check`, `sec_scan`, `migration_apply`).
      The first two are deterministic checks - a re-run reaches the same
      verdict, so failing *is* the correct outcome and a retry would only
      spend time to reach the same halt. `migration_apply` runs
      `ef.database_update` against a real database because a human approved
      that specific SQL; re-running it automatically would repeat an approved
      write under no decision (CLAUDE.md section 7, "non-idempotent
      operations are not blindly retried").
    - **`repair`** is itself the recovery position for `test_run`. A repair
      that failed must stop the cycle rather than restart itself, which is
      why `repair -> test_run` is `on_success` and not `always`."""
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")

    # Forward reachability only: a gate's `on_rejected` edge points back at
    # its own producer, which is upstream of the approval, not downstream.
    downstream: set[str] = set()
    frontier = [n.id for n in graph.nodes if n.requires_approval]
    while frontier:
        for edge in graph.successors(frontier.pop()):
            if edge.condition is EdgeCondition.ON_REJECTED or edge.target in downstream:
                continue
            downstream.add(edge.target)
            frontier.append(edge.target)

    assert {"scaffold", "decompose", "migration", "test_gen", "review", "release"} <= downstream

    unrecoverable = sorted(
        n.id
        for n in graph.nodes
        if n.kind.value == "agent"
        and n.id in downstream
        and n.id != "repair"
        and not any(
            e.condition in (EdgeCondition.ON_FAILURE, EdgeCondition.ALWAYS)
            for e in graph.successors(n.id)
        )
    )
    assert unrecoverable == [], (
        f"{unrecoverable} run after a human approval and end the run on a single failure, "
        "discarding decisions that cannot be re-derived"
    )
