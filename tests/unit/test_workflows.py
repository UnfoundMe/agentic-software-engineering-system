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


def test_greenfield_workflow_has_bounded_per_task_build_repair_cycles() -> None:
    """docs/02 Phase 4: "dotnet build as an exit gate on every implementation
    task" - each has its own bounded repair cycle, distinct from test_run's."""
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    assert graph.node("build_domain").cycle_budget is not None
    assert graph.node("build_api").cycle_budget is not None


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
    these edges were added without also setting `join: any`."""
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    for node_id in ("arch", "migration", "release"):
        assert graph.node(node_id).join is JoinPolicy.ANY, node_id
