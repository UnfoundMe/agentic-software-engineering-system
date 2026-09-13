"""The declarative guardrail engine (docs/05 section 3.6) and per-actor
capability manifests (docs/05 section 3.3's `CapabilityManifest`).

Two independent mechanisms, deliberately kept separate rather than merged
into one:

- `PolicyEngine` evaluates an `Action` against YAML rule packs (change
  control, security, compliance, autonomy) - global governance, authored by
  a human, never by an agent.
- `CapabilityManifest` is what one actor (in Phase 4, one agent) is granted -
  an allowlist of tool names, writable paths and a budget. Requesting
  anything outside it is a self-escalation attempt, and `enforce_capability`
  is what turns that into a `PolicyEvaluation` a caller can log as a
  `POLICY_VIOLATION` event, exactly the failure mode CLAUDE.md section 3
  ("Agents cannot modify their own permissions") names directly.

Both return a `PolicyEvaluation`, never raise and never touch the event
store - the same pure-evaluator shape as `kernel.gates.BudgetEntryGate`, so a
caller composes them (`combine`) and decides what event to emit, exactly as
the scheduler already does for gate results today.

A policy pack or a capability manifest can only make a decision *stricter*
than a tool's own declared `requires_approval` floor, never looser - the
corollary of "agents cannot modify their own permissions" is that a YAML
pack cannot loosen them either.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ases.kernel.events import ActorKind
from ases.kernel.tools.classification import SideEffect


class PolicyDecision(StrEnum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


_DECISION_RANK: Mapping[PolicyDecision, int] = {
    PolicyDecision.ALLOW: 0,
    PolicyDecision.REQUIRE_APPROVAL: 1,
    PolicyDecision.DENY: 2,
}


@dataclass(frozen=True)
class Action:
    """One tool invocation, described structurally - Phase 3 has no other
    kind of side-effecting action to evaluate yet."""

    tool_name: str
    side_effect: SideEffect
    actor_kind: ActorKind
    requires_approval: bool  # the tool's own declared floor (`ToolSpec.requires_approval`)


@dataclass(frozen=True)
class PolicyContext:
    run_id: str
    node_id: str | None = None


@dataclass(frozen=True)
class PolicyEvaluation:
    decision: PolicyDecision
    reason: str
    matched_rule: str | None = None

    @property
    def denied(self) -> bool:
        return self.decision is PolicyDecision.DENY


def combine(evaluations: Iterable[PolicyEvaluation]) -> PolicyEvaluation:
    """The strictest of several evaluations wins - never the loosest."""
    best = PolicyEvaluation(decision=PolicyDecision.ALLOW, reason="no restriction applied")
    for evaluation in evaluations:
        if _DECISION_RANK[evaluation.decision] > _DECISION_RANK[best.decision]:
            best = evaluation
    return best


@dataclass(frozen=True)
class PolicyRule:
    id: str
    description: str
    decision: PolicyDecision
    reason: str
    when_tool_name: str | None = None
    when_side_effect: SideEffect | None = None
    when_actor_kind: ActorKind | None = None

    def matches(self, action: Action) -> bool:
        if self.when_tool_name is not None and self.when_tool_name != action.tool_name:
            return False
        if self.when_side_effect is not None and self.when_side_effect != action.side_effect:
            return False
        return not (self.when_actor_kind is not None and self.when_actor_kind != action.actor_kind)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PolicyRule:
        side_effect = raw.get("when_side_effect")
        actor_kind = raw.get("when_actor_kind")
        return cls(
            id=str(raw["id"]),
            description=str(raw.get("description", "")),
            decision=PolicyDecision(raw["decision"]),
            reason=str(raw["reason"]),
            when_tool_name=raw.get("when_tool_name"),
            when_side_effect=SideEffect(side_effect) if side_effect is not None else None,
            when_actor_kind=ActorKind(actor_kind) if actor_kind is not None else None,
        )


def load_policy_rules(paths: Iterable[Path]) -> tuple[PolicyRule, ...]:
    """Loads and concatenates every rule pack in `paths`, in order. A rule
    pack is a YAML mapping with a top-level `rules:` list - see
    `policies/*.yaml` for the four named packs (docs/02 Phase 3: change
    control, security, compliance, autonomy)."""
    rules: list[PolicyRule] = []
    for path in paths:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for raw_rule in document.get("rules", []):
            rules.append(PolicyRule.from_dict(raw_rule))
    return tuple(rules)


class PolicyEngine:
    """Evaluates an `Action` against loaded rule packs, floored by the
    tool's own `requires_approval`."""

    def __init__(self, rules: tuple[PolicyRule, ...] = ()) -> None:
        self._rules = rules

    @property
    def rules(self) -> tuple[PolicyRule, ...]:
        return self._rules

    def evaluate(self, action: Action, ctx: PolicyContext) -> PolicyEvaluation:
        if action.requires_approval:
            floor = PolicyEvaluation(
                decision=PolicyDecision.REQUIRE_APPROVAL,
                reason="tool's own declared requires_approval",
            )
        else:
            floor = PolicyEvaluation(decision=PolicyDecision.ALLOW, reason="no restriction applied")
        matches = (
            PolicyEvaluation(decision=rule.decision, reason=rule.reason, matched_rule=rule.id)
            for rule in self._rules
            if rule.matches(action)
        )
        return combine((floor, *matches))


class CapabilityManifest(BaseModel):
    """What one actor may do: an allowlist, not a denylist - a tool or path
    not named here is refused, matching deny-by-default (docs/05 section
    3.3: `capabilities: ClassVar[CapabilityManifest]  # tools, writable
    paths, budget`)."""

    model_config = ConfigDict(frozen=True)

    actor: str
    allowed_tools: frozenset[str] = Field(default_factory=frozenset)
    writable_paths: tuple[str, ...] = ()
    max_tokens: int | None = None
    max_usd: float | None = None

    def grants(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tools


def enforce_capability(manifest: CapabilityManifest, action: Action) -> PolicyEvaluation:
    """A request for a tool the manifest does not grant is a self-escalation
    attempt (CLAUDE.md section 3), reported as `DENY` - never silently
    dropped, and never something the requesting actor's own manifest could
    have permitted regardless of its content, since the check is external to
    it."""
    if manifest.grants(action.tool_name):
        return PolicyEvaluation(
            decision=PolicyDecision.ALLOW, reason="granted by capability manifest"
        )
    return PolicyEvaluation(
        decision=PolicyDecision.DENY,
        reason=(
            f"actor {manifest.actor!r} attempted to use tool {action.tool_name!r}, "
            "which its capability manifest does not grant"
        ),
        matched_rule="capability_manifest",
    )


__all__ = [
    "Action",
    "CapabilityManifest",
    "PolicyContext",
    "PolicyDecision",
    "PolicyEngine",
    "PolicyEvaluation",
    "PolicyRule",
    "combine",
    "enforce_capability",
    "load_policy_rules",
]
