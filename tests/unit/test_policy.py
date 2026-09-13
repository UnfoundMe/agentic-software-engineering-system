"""`kernel.policy`: the declarative guardrail engine and capability manifests
(docs/05 section 3.6, docs/02 Phase 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ases.kernel.events import ActorKind
from ases.kernel.policy import (
    Action,
    CapabilityManifest,
    PolicyContext,
    PolicyDecision,
    PolicyEngine,
    PolicyEvaluation,
    PolicyRule,
    combine,
    enforce_capability,
    load_policy_rules,
)
from ases.kernel.tools.classification import SideEffect

_CTX = PolicyContext(run_id="run-1")


def _action(**overrides: object) -> Action:
    defaults: dict[str, object] = {
        "tool_name": "fs.write_file",
        "side_effect": SideEffect.SANDBOX,
        "actor_kind": ActorKind.AGENT,
        "requires_approval": False,
    }
    defaults.update(overrides)
    return Action(**defaults)  # type: ignore[arg-type]


# -- PolicyEngine basics -----------------------------------------------------


def test_an_action_with_no_matching_rule_and_no_approval_floor_is_allowed() -> None:
    engine = PolicyEngine()
    result = engine.evaluate(_action(), _CTX)
    assert result.decision is PolicyDecision.ALLOW


def test_a_tools_own_requires_approval_floor_is_respected_with_no_rules() -> None:
    engine = PolicyEngine()
    result = engine.evaluate(_action(requires_approval=True), _CTX)
    assert result.decision is PolicyDecision.REQUIRE_APPROVAL


def test_a_matching_rule_denies() -> None:
    rule = PolicyRule(
        id="deny-writes",
        description="",
        decision=PolicyDecision.DENY,
        reason="test",
        when_tool_name="fs.write_file",
    )
    engine = PolicyEngine((rule,))
    result = engine.evaluate(_action(tool_name="fs.write_file"), _CTX)
    assert result.decision is PolicyDecision.DENY
    assert result.matched_rule == "deny-writes"


def test_a_non_matching_rule_does_not_apply() -> None:
    rule = PolicyRule(
        id="deny-other",
        description="",
        decision=PolicyDecision.DENY,
        reason="test",
        when_tool_name="git.diff",
    )
    engine = PolicyEngine((rule,))
    result = engine.evaluate(_action(tool_name="fs.write_file"), _CTX)
    assert result.decision is PolicyDecision.ALLOW


def test_a_rule_can_only_tighten_never_loosen_a_tools_own_approval_floor() -> None:
    """A policy pack that would ALLOW something the tool itself flags
    requires_approval must not be able to loosen it - the floor always wins
    when it is stricter than what any rule proposes."""
    rule = PolicyRule(
        id="explicit-allow",
        description="",
        decision=PolicyDecision.ALLOW,
        reason="test",
        when_tool_name="fs.write_file",
    )
    engine = PolicyEngine((rule,))
    result = engine.evaluate(_action(tool_name="fs.write_file", requires_approval=True), _CTX)
    assert result.decision is PolicyDecision.REQUIRE_APPROVAL


def test_multiple_matching_rules_take_the_strictest() -> None:
    rules = (
        PolicyRule(
            id="require-approval",
            description="",
            decision=PolicyDecision.REQUIRE_APPROVAL,
            reason="test",
            when_side_effect=SideEffect.SANDBOX,
        ),
        PolicyRule(
            id="deny",
            description="",
            decision=PolicyDecision.DENY,
            reason="test",
            when_tool_name="fs.write_file",
        ),
    )
    engine = PolicyEngine(rules)
    result = engine.evaluate(_action(), _CTX)
    assert result.decision is PolicyDecision.DENY


def test_when_side_effect_matches_only_that_side_effect_class() -> None:
    rule = PolicyRule(
        id="external-approval",
        description="",
        decision=PolicyDecision.REQUIRE_APPROVAL,
        reason="test",
        when_side_effect=SideEffect.EXTERNAL,
    )
    engine = PolicyEngine((rule,))
    sandbox_result = engine.evaluate(_action(side_effect=SideEffect.SANDBOX), _CTX)
    external_result = engine.evaluate(_action(side_effect=SideEffect.EXTERNAL), _CTX)
    assert sandbox_result.decision is PolicyDecision.ALLOW
    assert external_result.decision is PolicyDecision.REQUIRE_APPROVAL


def test_when_actor_kind_matches_only_that_actor() -> None:
    rule = PolicyRule(
        id="tools-cannot-invoke-tools",
        description="",
        decision=PolicyDecision.DENY,
        reason="test",
        when_actor_kind=ActorKind.TOOL,
    )
    engine = PolicyEngine((rule,))
    agent_result = engine.evaluate(_action(actor_kind=ActorKind.AGENT), _CTX)
    tool_result = engine.evaluate(_action(actor_kind=ActorKind.TOOL), _CTX)
    assert agent_result.decision is PolicyDecision.ALLOW
    assert tool_result.decision is PolicyDecision.DENY


# -- combine ------------------------------------------------------------------


def test_combine_of_nothing_is_allow() -> None:
    assert combine(()).decision is PolicyDecision.ALLOW


def test_combine_picks_the_strictest_of_several() -> None:
    evaluations = [
        PolicyEvaluation(decision=PolicyDecision.ALLOW, reason="a"),
        PolicyEvaluation(decision=PolicyDecision.REQUIRE_APPROVAL, reason="b"),
        PolicyEvaluation(decision=PolicyDecision.ALLOW, reason="c"),
    ]
    assert combine(evaluations).decision is PolicyDecision.REQUIRE_APPROVAL


# -- CapabilityManifest / self-escalation ------------------------------------


def test_a_granted_tool_is_allowed() -> None:
    manifest = CapabilityManifest(actor="requirements", allowed_tools=frozenset({"fs.write_file"}))
    result = enforce_capability(manifest, _action(tool_name="fs.write_file"))
    assert result.decision is PolicyDecision.ALLOW


def test_requesting_a_tool_outside_the_manifest_is_a_self_escalation_denied() -> None:
    manifest = CapabilityManifest(actor="requirements", allowed_tools=frozenset({"git.status"}))
    result = enforce_capability(manifest, _action(tool_name="dotnet.build"))
    assert result.decision is PolicyDecision.DENY
    assert result.matched_rule == "capability_manifest"
    assert "requirements" in result.reason
    assert "dotnet.build" in result.reason


def test_an_empty_manifest_grants_nothing() -> None:
    manifest = CapabilityManifest(actor="bare")
    result = enforce_capability(manifest, _action(tool_name="git.status"))
    assert result.decision is PolicyDecision.DENY


def test_capability_manifest_is_frozen() -> None:
    manifest = CapabilityManifest(actor="x")
    with pytest.raises(Exception):
        manifest.actor = "y"  # type: ignore[misc]


# -- YAML rule pack loading ---------------------------------------------------

_POLICIES_DIR = Path(__file__).parents[2] / "src/orchestrator/ases/policies"


def test_all_four_named_packs_load_without_error() -> None:
    paths = [
        _POLICIES_DIR / "change_control.yaml",
        _POLICIES_DIR / "security.yaml",
        _POLICIES_DIR / "compliance.yaml",
        _POLICIES_DIR / "autonomy.yaml",
    ]
    rules = load_policy_rules(paths)
    assert isinstance(rules, tuple)


def test_compliance_pack_is_intentionally_empty() -> None:
    rules = load_policy_rules([_POLICIES_DIR / "compliance.yaml"])
    assert rules == ()


def test_change_control_pack_requires_approval_for_external_side_effects() -> None:
    rules = load_policy_rules([_POLICIES_DIR / "change_control.yaml"])
    engine = PolicyEngine(rules)
    result = engine.evaluate(_action(side_effect=SideEffect.EXTERNAL), _CTX)
    assert result.decision is PolicyDecision.REQUIRE_APPROVAL


def test_security_pack_requires_approval_for_dotnet_new() -> None:
    rules = load_policy_rules([_POLICIES_DIR / "security.yaml"])
    engine = PolicyEngine(rules)
    result = engine.evaluate(_action(tool_name="dotnet.new", side_effect=SideEffect.SANDBOX), _CTX)
    assert result.decision is PolicyDecision.REQUIRE_APPROVAL


def test_autonomy_pack_denies_a_tool_actor() -> None:
    rules = load_policy_rules([_POLICIES_DIR / "autonomy.yaml"])
    engine = PolicyEngine(rules)
    result = engine.evaluate(_action(actor_kind=ActorKind.TOOL), _CTX)
    assert result.decision is PolicyDecision.DENY


def test_loading_all_four_packs_together_combines_their_rules() -> None:
    paths = [
        _POLICIES_DIR / "change_control.yaml",
        _POLICIES_DIR / "security.yaml",
        _POLICIES_DIR / "compliance.yaml",
        _POLICIES_DIR / "autonomy.yaml",
    ]
    engine = PolicyEngine(load_policy_rules(paths))
    # From security.yaml:
    assert (
        engine.evaluate(_action(tool_name="dotnet.new"), _CTX).decision
        is PolicyDecision.REQUIRE_APPROVAL
    )
    # From autonomy.yaml:
    assert engine.evaluate(_action(actor_kind=ActorKind.TOOL), _CTX).decision is PolicyDecision.DENY
    # Unmatched by any pack:
    assert (
        engine.evaluate(_action(tool_name="git.status", side_effect=SideEffect.NONE), _CTX).decision
        is PolicyDecision.ALLOW
    )
