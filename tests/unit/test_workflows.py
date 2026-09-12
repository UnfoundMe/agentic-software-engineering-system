"""The checked-in workflow YAML files load and validate."""

from __future__ import annotations

from pathlib import Path

import ases
from ases.kernel.graph import WorkflowGraph

WORKFLOWS_DIR = Path(ases.__file__).parent / "workflows"


def test_greenfield_workflow_loads_and_validates() -> None:
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    assert graph.name == "greenfield"
    assert graph.entry == ("req",)
    assert "summary" in graph.terminal_nodes


def test_greenfield_workflow_has_the_documented_gates() -> None:
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    gate_ids = {n.id for n in graph.nodes if n.kind.value == "gate" and n.requires_approval}
    assert gate_ids == {"gate1", "gate2", "gate3"}


def test_greenfield_workflow_has_a_bounded_repair_cycle() -> None:
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    test_run = graph.node("test_run")
    assert test_run.cycle_budget is not None


def test_greenfield_workflow_has_a_bounded_clarification_cycle() -> None:
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    req = graph.node("req")
    assert req.cycle_budget is not None
