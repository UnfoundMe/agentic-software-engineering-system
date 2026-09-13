"""Scoped context retrieval (docs/05 section 3.3)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from ases.context.lineage import UnknownArtifactError
from ases.context.retriever import ArtifactContentUnavailableError, ArtifactRef, ContextRetriever
from ases.kernel.state import ArtifactRecord, NodeState, RunState


def _state_with_one_artifact() -> RunState:
    state = RunState(run_id=uuid4())
    record = ArtifactRecord(
        artifact_hash="abc123",
        kind="RequirementSpec",
        node_id="req",
        produced_at=datetime.now(UTC),
        inputs=("upstream1", "upstream2"),
        validated=True,
    )
    state.artifacts["abc123"] = record
    state.nodes["req"] = NodeState(node_id="req", produced=("abc123",))
    return state


def test_summary_reports_metadata_without_content() -> None:
    retriever = ContextRetriever(_state_with_one_artifact())
    summary = retriever.summary("abc123")
    assert summary.artifact_hash == "abc123"
    assert summary.kind == "RequirementSpec"
    assert summary.node_id == "req"
    assert summary.validated is True
    assert summary.input_count == 2


def test_summary_of_unknown_hash_raises() -> None:
    retriever = ContextRetriever(_state_with_one_artifact())
    with pytest.raises(UnknownArtifactError):
        retriever.summary("does-not-exist")


def test_summaries_for_returns_everything_a_node_produced() -> None:
    retriever = ContextRetriever(_state_with_one_artifact())
    summaries = retriever.summaries_for("req")
    assert len(summaries) == 1
    assert summaries[0].artifact_hash == "abc123"


def test_summaries_for_an_unknown_node_is_empty_not_an_error() -> None:
    retriever = ContextRetriever(_state_with_one_artifact())
    assert retriever.summaries_for("nonexistent-node") == ()


def test_fetch_of_a_known_artifact_raises_content_unavailable() -> None:
    """No content-addressed store exists yet - see the module docstring. This
    must fail loudly, not return fabricated content."""
    retriever = ContextRetriever(_state_with_one_artifact())
    with pytest.raises(ArtifactContentUnavailableError):
        retriever.fetch(ArtifactRef(artifact_hash="abc123"))


def test_fetch_of_an_unknown_artifact_raises_unknown_artifact_not_content_unavailable() -> None:
    """The two failure modes are distinguishable: an agent asking for content
    that could never exist gets a different error than one asking for content
    this run legitimately produced but cannot yet retrieve."""
    retriever = ContextRetriever(_state_with_one_artifact())
    with pytest.raises(UnknownArtifactError):
        retriever.fetch(ArtifactRef(artifact_hash="does-not-exist"))


def test_retriever_never_needs_an_event_store() -> None:
    """Structural check for the class's whole reason to exist: it is
    constructed from a `RunState` alone."""
    import inspect

    params = list(inspect.signature(ContextRetriever.__init__).parameters)
    assert params == ["self", "state"]
