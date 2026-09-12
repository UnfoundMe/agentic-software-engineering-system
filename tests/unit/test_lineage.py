"""LineageGraph: forward and backward walks over RunState.artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from ases.context.lineage import LineageGraph, UnknownArtifactError
from ases.kernel.state import ArtifactRecord, RunState


def _record(
    artifact_hash: str, *, inputs: tuple[str, ...] = (), node_id: str = "n"
) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_hash=artifact_hash,
        kind="Test",
        node_id=node_id,
        produced_at=datetime.now(UTC),
        inputs=inputs,
    )


def _state_with(*records: ArtifactRecord) -> RunState:
    state = RunState(run_id=uuid4())
    for r in records:
        state.artifacts[r.artifact_hash] = r
    return state


def test_ancestors_of_a_root_artifact_is_empty() -> None:
    state = _state_with(_record("h1"))
    lineage = LineageGraph.from_state(state)
    assert lineage.ancestors("h1") == ()


def test_ancestors_walks_backward_through_a_chain() -> None:
    state = _state_with(
        _record("h1", node_id="req"),
        _record("h2", node_id="arch", inputs=("h1",)),
        _record("h3", node_id="impl", inputs=("h2",)),
    )
    lineage = LineageGraph.from_state(state)
    ancestor_hashes = [r.artifact_hash for r in lineage.ancestors("h3")]
    assert ancestor_hashes == ["h2", "h1"]


def test_descendants_walks_forward_through_a_chain() -> None:
    state = _state_with(
        _record("h1", node_id="req"),
        _record("h2", node_id="arch", inputs=("h1",)),
        _record("h3", node_id="impl", inputs=("h2",)),
    )
    lineage = LineageGraph.from_state(state)
    descendant_hashes = {r.artifact_hash for r in lineage.descendants("h1")}
    assert descendant_hashes == {"h2", "h3"}


def test_descendants_of_a_leaf_is_empty() -> None:
    state = _state_with(_record("h1"), _record("h2", inputs=("h1",)))
    lineage = LineageGraph.from_state(state)
    assert lineage.descendants("h2") == ()


def test_diamond_shaped_lineage_has_no_duplicates() -> None:
    """h1 feeds both h2 and h3, which both feed h4 - h1 must appear once in
    h4's ancestors, not twice."""
    state = _state_with(
        _record("h1"),
        _record("h2", inputs=("h1",)),
        _record("h3", inputs=("h1",)),
        _record("h4", inputs=("h2", "h3")),
    )
    lineage = LineageGraph.from_state(state)
    ancestor_hashes = [r.artifact_hash for r in lineage.ancestors("h4")]
    assert sorted(ancestor_hashes) == ["h1", "h2", "h3"]
    assert ancestor_hashes.count("h1") == 1


def test_affected_nodes_reduces_descendants_to_node_ids() -> None:
    state = _state_with(
        _record("h1", node_id="req"),
        _record("h2", node_id="arch", inputs=("h1",)),
        _record("h3", node_id="impl", inputs=("h2",)),
    )
    lineage = LineageGraph.from_state(state)
    assert lineage.affected_nodes("h1") == frozenset({"arch", "impl"})


def test_unknown_artifact_raises() -> None:
    lineage = LineageGraph.from_state(_state_with(_record("h1")))
    with pytest.raises(UnknownArtifactError):
        lineage.ancestors("does-not-exist")
    with pytest.raises(UnknownArtifactError):
        lineage.descendants("does-not-exist")


def test_input_from_outside_this_runs_artifacts_is_tolerated() -> None:
    """An artifact whose recorded input hash was never itself produced in this
    run (e.g. an external baseline) must not crash the backward walk."""
    state = _state_with(_record("h2", inputs=("external-hash",)))
    lineage = LineageGraph.from_state(state)
    assert lineage.ancestors("h2") == ()
