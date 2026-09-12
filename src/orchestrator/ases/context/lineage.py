"""The provenance DAG: why an artifact exists, and what depends on it.

Built entirely from `RunState.artifacts`, which already carries each
artifact's `inputs` (populated by the scheduler from the producing node's
predecessors' latest output). This module adds no new data - it only answers
two graph-shaped questions over data the kernel's fold already captured:

- **Backward** (`ancestors`): why does this artifact exist? Walk `inputs` back
  to the artifacts and, transitively, the human decisions that produced them.
- **Forward** (`descendants`): what would `STALE` if this artifact changed?
  Walk the reverse edges to find everything that consumed it, directly or not.

This is deliberately a one-way dependency: it imports `ases.kernel.state`,
never the reverse. `kernel/` stays free to run without this module ever being
imported, matching the layering rule - lineage is something built *from* a
fold, not something the fold depends on.

Phase 7's `replan` module is the consumer that turns `descendants()` into
actual `NODE_MARKED_STALE` / `APPROVAL_REVOKED` events; this module only
answers the graph question.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from ases.kernel.state import ArtifactRecord, RunState


class UnknownArtifactError(KeyError):
    """The requested artifact hash is not in this run's state."""


class LineageGraph(BaseModel):
    """A read-only view over one run's artifact provenance."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    records: Mapping[str, ArtifactRecord]
    #: Reverse edges: artifact_hash -> hashes of artifacts whose `inputs` name it.
    #: Precomputed once at construction, since `descendants` is the walk this
    #: module exists for and `ArtifactRecord.inputs` only stores the forward
    #: (consumer -> producer) direction.
    consumers: Mapping[str, tuple[str, ...]]

    @classmethod
    def from_state(cls, state: RunState) -> LineageGraph:
        consumers: dict[str, list[str]] = {h: [] for h in state.artifacts}
        for record in state.artifacts.values():
            for input_hash in record.inputs:
                consumers.setdefault(input_hash, []).append(record.artifact_hash)
        return cls(
            records=dict(state.artifacts),
            consumers={k: tuple(v) for k, v in consumers.items()},
        )

    def _require(self, artifact_hash: str) -> ArtifactRecord:
        try:
            return self.records[artifact_hash]
        except KeyError:
            raise UnknownArtifactError(artifact_hash) from None

    def ancestors(self, artifact_hash: str) -> tuple[ArtifactRecord, ...]:
        """Every artifact that transitively fed into `artifact_hash`.

        Ordered by discovery (breadth-first), nearest first - useful for a
        "why does this exist" trail where the immediate cause matters more
        than the root requirement.
        """
        self._require(artifact_hash)
        seen: set[str] = {artifact_hash}
        ordered: list[ArtifactRecord] = []
        frontier = list(self.records[artifact_hash].inputs)
        while frontier:
            next_frontier: list[str] = []
            for h in frontier:
                if h in seen:
                    continue
                seen.add(h)
                record = self.records.get(h)
                if record is None:
                    continue  # an input hash from outside this run's artifacts
                ordered.append(record)
                next_frontier.extend(record.inputs)
            frontier = next_frontier
        return tuple(ordered)

    def descendants(self, artifact_hash: str) -> tuple[ArtifactRecord, ...]:
        """Every artifact that transitively consumed `artifact_hash`.

        This is the set Phase 7 marks `STALE` when `artifact_hash`'s producing
        node is amended and re-run.
        """
        self._require(artifact_hash)
        seen: set[str] = {artifact_hash}
        ordered: list[ArtifactRecord] = []
        frontier = list(self.consumers.get(artifact_hash, ()))
        while frontier:
            next_frontier: list[str] = []
            for h in frontier:
                if h in seen:
                    continue
                seen.add(h)
                record = self.records[h]
                ordered.append(record)
                next_frontier.extend(self.consumers.get(h, ()))
            frontier = next_frontier
        return tuple(ordered)

    def affected_nodes(self, artifact_hash: str) -> frozenset[str]:
        """Convenience: the node ids that would need to re-run if
        `artifact_hash` changed - `descendants` reduced to node identity."""
        return frozenset(record.node_id for record in self.descendants(artifact_hash))
