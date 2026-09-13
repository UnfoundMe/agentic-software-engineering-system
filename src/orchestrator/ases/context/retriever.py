"""Scoped context retrieval (docs/05 section 3.3).

An agent must not receive the full run history - by the time a late-stage
agent (release readiness, docs) runs, the accumulated log of every prior
node's reasoning is unreadable and drives exactly the "late-stage drift" this
module exists to prevent. Instead an agent is handed a `ContextRetriever`
scoped to the current run's folded state, and asks for what it actually
needs: `summaries_for` and `summary` return metadata cheaply, and full
content is fetched explicitly and only for artifacts named by hash.

**A gap this module used to have, closed as part of Phase 4:** `fetch()`
previously always raised `ArtifactContentUnavailableError`, because
`ARTIFACT_PRODUCED` only ever recorded an artifact's hash, never its content
(docs/05 section 8 talks about content-addressing, but no code path wrote the
payload anywhere retrievable). Phase 4 is the first phase with agents that
actually need to read a previous node's real output, so
`kernel.scheduler._finish_agent_node` now includes the full payload in the
event (`content=...`), and `kernel.state.RunState.artifact_content` captures
it during the fold. `fetch()` below reads that and validates it back into the
artifact's declared contract type via `ases.contracts.CONTRACTS`. This is
still not a separate artifact store (`docs/02`'s `context/store.py` remains
unbuilt) - the event log stays the only place content lives, matching "state
is a fold" - and `fetch()` still raises `ArtifactContentUnavailableError` for
the one real remaining case: a hash that predates this change (replayed from
an older, content-less export).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from ases.context.lineage import UnknownArtifactError
from ases.contracts import CONTRACTS
from ases.kernel.state import ArtifactRecord, RunState


class ArtifactRef(BaseModel):
    """Names one artifact an agent wants full content for."""

    model_config = ConfigDict(frozen=True)

    artifact_hash: str


class NoArtifactFromNodeError(LookupError):
    """`fetch_latest_from` was asked for a node that has not (yet) produced
    anything - either an authoring bug (the wrong node id) or a real
    graph-ordering bug (the caller ran before its declared upstream node)."""

    def __init__(self, node_id: str) -> None:
        super().__init__(f"node {node_id!r} has not produced any artifact yet")
        self.node_id = node_id


class UnknownArtifactKindError(KeyError):
    """The artifact's recorded `kind` has no entry in `ases.contracts.CONTRACTS` -
    a workflow YAML named a `produces:` kind with no matching contract model,
    which `tests/unit/test_contracts.py` is meant to catch before a run ever
    reaches this point."""


class ArtifactContentUnavailableError(RuntimeError):
    """No content-addressed store exists yet to resolve this hash to a
    payload. See module docstring."""

    def __init__(self, artifact_hash: str) -> None:
        super().__init__(
            f"artifact {artifact_hash} has no retrievable content: no content-addressed "
            "artifact store exists yet (docs/02's context/store.py is unbuilt). Only "
            "summaries are available until Phase 4 wires one."
        )
        self.artifact_hash = artifact_hash


class ArtifactSummary(BaseModel):
    """What an agent gets by default: enough to know an artifact exists and
    what produced it, without its full content."""

    model_config = ConfigDict(frozen=True)

    artifact_hash: str
    kind: str
    node_id: str
    validated: bool
    input_count: int

    @classmethod
    def _from_record(cls, record: ArtifactRecord) -> ArtifactSummary:
        return cls(
            artifact_hash=record.artifact_hash,
            kind=record.kind,
            node_id=record.node_id,
            validated=record.validated,
            input_count=len(record.inputs),
        )


class ContextRetriever:
    """Scoped view over one run's folded state, for agent consumption.

    Deliberately holds a `RunState`, never an `EventStore` - an agent that can
    reach the store could read another run's history or replay events itself,
    which is exactly the unscoped access this class exists to prevent.
    """

    def __init__(self, state: RunState) -> None:
        self._state = state

    def summary(self, artifact_hash: str) -> ArtifactSummary:
        record = self._state.artifacts.get(artifact_hash)
        if record is None:
            raise UnknownArtifactError(artifact_hash)
        return ArtifactSummary._from_record(record)

    def summaries_for(self, node_id: str) -> tuple[ArtifactSummary, ...]:
        """Every artifact `node_id` has produced, most recent last - the
        order `NodeState.produced` already records them in."""
        node = self._state.nodes.get(node_id)
        if node is None:
            return ()
        return tuple(self.summary(h) for h in node.produced)

    def fetch(self, ref: ArtifactRef) -> BaseModel:
        """Full artifact content, validated back into its declared contract
        type. Raises `ArtifactContentUnavailableError` only for the residual
        case described in the module docstring - a hash from before this
        capability existed."""
        record = self._state.artifacts.get(ref.artifact_hash)
        if record is None:
            raise UnknownArtifactError(ref.artifact_hash)
        content: Any = self._state.artifact_content.get(ref.artifact_hash)
        if content is None:
            raise ArtifactContentUnavailableError(ref.artifact_hash)
        contract = CONTRACTS.get(record.kind)
        if contract is None:
            raise UnknownArtifactKindError(record.kind)
        return contract.model_validate(content)

    def fetch_latest_from(self, node_id: str) -> BaseModel:
        """Convenience for the common case every downstream agent needs:
        "give me the current content of the artifact my known upstream node
        produced." A workflow's node choreography (docs/03 section 3.3) is a
        fixed contract between adjacent agents - each agent already knows
        which node id feeds it, so this is `summaries_for(node_id)[-1]` plus
        `fetch(...)` in one call, not a generic graph traversal."""
        summaries = self.summaries_for(node_id)
        if not summaries:
            raise NoArtifactFromNodeError(node_id)
        return self.fetch(ArtifactRef(artifact_hash=summaries[-1].artifact_hash))

    def rejection_reason_of(self, node_id: str) -> str | None:
        """The human's stated reason the last time `node_id` (a gate) was
        rejected - `None` if it was never rejected, was granted, or does not
        exist. Needed by whichever agent a gate's `on_rejected` edge loops
        back to, so a retry actually responds to *why* a human rejected it
        rather than blindly reproducing the same output (docs/05 section
        7.1's "compiler diagnostics fed back verbatim" principle, applied to
        a human's rejection instead of a tool's failure).

        Deliberately keyed by the *gate's* node id, not the agent's own -
        `RunState.approvals` is recorded per approval-requesting node, and a
        gate is always a distinct node from the agent whose output it
        approves (see `workflows/greenfield.yaml`'s `gate1`/`gate2`/
        `migration_gate`/`gate3`)."""
        record = self._state.approvals.get(node_id)
        if record is None or record.granted:
            return None
        return record.reason

    def last_error_of(self, node_id: str) -> str | None:
        """The verbatim error a node's most recent failed attempt recorded -
        `None` if the node has never failed (or does not exist yet).

        Needed by a repair agent to see *why* a prior attempt failed (docs/05
        section 7.1: "compiler diagnostics fed back verbatim"). No new
        mechanism: `kernel.state.NodeState.last_error` already captures this
        from `NODE_FAILED` events as part of Phase 1's fold - this only
        exposes it through the same scoped-access seam every other read
        here goes through, rather than an agent reaching into `RunState`
        directly.
        """
        node = self._state.nodes.get(node_id)
        return node.last_error if node is not None else None
