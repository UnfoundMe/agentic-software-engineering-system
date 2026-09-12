"""Shared base for every artifact contract.

Every agent output validates against one of these models before an
`ARTIFACT_PRODUCED` event is trusted (L1 schema validation, docs/05 section 6).
Agents do not exist yet (Phase 4); these contracts are written now because
docs/03 already names every artifact kind precisely enough to draft them, and
because a stable target is what Phase 4's agents will be built against -
waiting for agents to exist first would get the sequencing backwards.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ArtifactModel(BaseModel):
    """Base for every typed artifact an agent can produce.

    `extra="forbid"` is deliberate: an agent's structured output either
    matches the contract or it doesn't. Silently accepting unknown fields
    would let a model drift away from the schema without anyone noticing -
    exactly the failure L1 validation exists to catch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
