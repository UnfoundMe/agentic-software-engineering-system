"""Scoped context retrieval (docs/05 section 3.3).

An agent must not receive the full run history - by the time a late-stage
agent (release readiness, docs) runs, the accumulated log of every prior
node's reasoning is unreadable and drives exactly the "late-stage drift" this
module exists to prevent. Instead an agent is handed a `ContextRetriever`
scoped to the current run's folded state, and asks for what it actually
needs: `summaries_for` and `summary` return metadata cheaply, and full
content is fetched explicitly and only for artifacts named by hash.

**Known gap, not silently worked around:** `ArtifactRecord` (kernel/state.py)
carries an artifact's hash, kind, producing node and inputs, but never its
content - the event log only ever records `ARTIFACT_PRODUCED`'s hash, by
design (docs/05 section 8: identical content is stored once). There is no
content-addressed artifact store yet mapping a hash back to the payload that
produced it - `docs/02`'s repository layout names `context/store.py` for this,
but no phase bullet builds it, and no code path today ever writes an
artifact's payload anywhere the hash could be resolved back to. `fetch()`
below raises `ArtifactContentUnavailableError` rather than inventing a
content-addressed store retrofitted onto the Phase 1 scheduler - that
plumbing belongs with Phase 4, once an actual agent has content worth
storing and a scheduler change is in scope. Summaries need no such store,
since everything they report already lives in `RunState`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ases.context.lineage import UnknownArtifactError
from ases.kernel.state import ArtifactRecord, RunState


class ArtifactRef(BaseModel):
    """Names one artifact an agent wants full content for."""

    model_config = ConfigDict(frozen=True)

    artifact_hash: str


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
        """Full artifact content. Always raises today - see
        `ArtifactContentUnavailableError` and the module docstring."""
        if ref.artifact_hash not in self._state.artifacts:
            raise UnknownArtifactError(ref.artifact_hash)
        raise ArtifactContentUnavailableError(ref.artifact_hash)
