"""Scoped context retrieval (docs/05 section 3.3)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from ases.context.lineage import UnknownArtifactError
from ases.context.retriever import (
    ArtifactContentUnavailableError,
    ArtifactRef,
    ContextRetriever,
    NoArtifactFromNodeError,
    UnknownArtifactKindError,
)
from ases.contracts.artifacts import RequirementSpec
from ases.kernel.state import ApprovalRecord, ArtifactRecord, NodeState, RunState


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


def test_fetch_of_a_known_artifact_with_no_recorded_content_raises() -> None:
    """A hash from before content-capture existed (or a tool-produced
    artifact whose payload was never a full contract) must fail loudly, not
    return fabricated content."""
    retriever = ContextRetriever(_state_with_one_artifact())
    with pytest.raises(ArtifactContentUnavailableError):
        retriever.fetch(ArtifactRef(artifact_hash="abc123"))


def test_fetch_returns_the_content_validated_into_its_declared_contract() -> None:
    state = _state_with_one_artifact()
    state.artifact_content["abc123"] = {"summary": "s", "source_text": "raw"}

    retriever = ContextRetriever(state)
    fetched = retriever.fetch(ArtifactRef(artifact_hash="abc123"))

    assert fetched == RequirementSpec(summary="s", source_text="raw")


def test_fetch_of_content_for_an_unrecognised_kind_raises() -> None:
    state = _state_with_one_artifact()
    state.artifacts["abc123"] = state.artifacts["abc123"].model_copy(
        update={"kind": "NotARealContractKind"}
    )
    state.artifact_content["abc123"] = {"whatever": "value"}

    retriever = ContextRetriever(state)
    with pytest.raises(UnknownArtifactKindError):
        retriever.fetch(ArtifactRef(artifact_hash="abc123"))


def test_fetch_of_an_unknown_artifact_raises_unknown_artifact_not_content_unavailable() -> None:
    """The two failure modes are distinguishable: an agent asking for content
    that could never exist gets a different error than one asking for content
    this run legitimately produced but cannot yet retrieve."""
    retriever = ContextRetriever(_state_with_one_artifact())
    with pytest.raises(UnknownArtifactError):
        retriever.fetch(ArtifactRef(artifact_hash="does-not-exist"))


def test_fetch_latest_from_returns_the_most_recently_produced_content() -> None:
    state = _state_with_one_artifact()
    state.artifact_content["abc123"] = {"summary": "s", "source_text": "raw"}

    retriever = ContextRetriever(state)
    fetched = retriever.fetch_latest_from("req")

    assert fetched == RequirementSpec(summary="s", source_text="raw")


def test_fetch_latest_from_picks_the_last_of_several_produced_artifacts() -> None:
    state = _state_with_one_artifact()
    state.artifact_content["abc123"] = {"summary": "first", "source_text": "raw"}
    second = ArtifactRecord(
        artifact_hash="def456",
        kind="RequirementSpec",
        node_id="req",
        produced_at=datetime.now(UTC),
    )
    state.artifacts["def456"] = second
    state.artifact_content["def456"] = {"summary": "second (repaired)", "source_text": "raw"}
    state.nodes["req"] = NodeState(node_id="req", produced=("abc123", "def456"))

    retriever = ContextRetriever(state)
    fetched = retriever.fetch_latest_from("req")

    assert fetched == RequirementSpec(summary="second (repaired)", source_text="raw")


def test_fetch_latest_from_a_node_with_nothing_produced_raises() -> None:
    retriever = ContextRetriever(_state_with_one_artifact())
    with pytest.raises(NoArtifactFromNodeError):
        retriever.fetch_latest_from("nonexistent-node")


def test_last_error_of_returns_the_recorded_failure_message() -> None:
    state = _state_with_one_artifact()
    state.nodes["req"].last_error = "CS0103: 'Foo' does not exist in the current context"

    retriever = ContextRetriever(state)

    assert retriever.last_error_of("req") == "CS0103: 'Foo' does not exist in the current context"


def test_last_error_of_a_node_that_never_failed_is_none() -> None:
    retriever = ContextRetriever(_state_with_one_artifact())
    assert retriever.last_error_of("req") is None


def test_last_error_of_an_unknown_node_is_none_not_an_error() -> None:
    retriever = ContextRetriever(_state_with_one_artifact())
    assert retriever.last_error_of("nonexistent-node") is None


def test_rejection_reason_of_returns_the_recorded_reason() -> None:
    state = _state_with_one_artifact()
    state.approvals["gate2"] = ApprovalRecord(
        node_id="gate2",
        artifact_hash="h",
        granted=False,
        actor="human:alice",
        decided_at=datetime.now(UTC),
        reason="the layering is wrong",
    )

    retriever = ContextRetriever(state)

    assert retriever.rejection_reason_of("gate2") == "the layering is wrong"


def test_rejection_reason_of_a_granted_gate_is_none() -> None:
    state = _state_with_one_artifact()
    state.approvals["gate2"] = ApprovalRecord(
        node_id="gate2",
        artifact_hash="h",
        granted=True,
        actor="human:alice",
        decided_at=datetime.now(UTC),
        reason="looks good",
    )

    retriever = ContextRetriever(state)

    assert retriever.rejection_reason_of("gate2") is None


def test_rejection_reason_of_a_gate_never_decided_is_none() -> None:
    retriever = ContextRetriever(_state_with_one_artifact())
    assert retriever.rejection_reason_of("gate2") is None


def test_retriever_never_needs_an_event_store() -> None:
    """Structural check for the class's whole reason to exist: it is
    constructed from a `RunState` alone."""
    import inspect

    params = list(inspect.signature(ContextRetriever.__init__).parameters)
    assert params == ["self", "state"]
