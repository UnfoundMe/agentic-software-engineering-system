"""Phase 3's own exit criterion (docs/02): "the invariants suite (§1) passes -
every violation attempt is refused and logged."

The individual mechanisms are already unit-tested in depth
(`tests/unit/test_tool_registry.py`, `test_policy.py`, `test_tool_fs.py`,
`test_sandbox_workspace.py`). This file exists to state each CLAUDE.md-level
invariant once, explicitly, in the same place `test_layering.py` states
invariants 1-3 - so a reviewer can see the whole Phase 3 governance surface
without piecing it together from individually-motivated unit tests.

"Refused and logged" here means: the attempt produces a typed, inspectable
result whose denial a caller can turn into a `POLICY_VIOLATION` or
`TOOL_FAILED` event - the same pure-evaluator contract every kernel gate
already uses (`kernel.gates.BudgetEntryGate`). Nothing in this suite invents
an event-emission mechanism kernel/tools and kernel/policy do not have; it
proves the refusal itself is real and structural, not that a specific event
was written to a store no test here has access to.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from ases.kernel.events import ActorKind
from ases.kernel.policy import (
    Action,
    CapabilityManifest,
    PolicyContext,
    PolicyDecision,
    PolicyEngine,
    PolicyRule,
    enforce_capability,
    load_policy_rules,
)
from ases.kernel.tools.classification import SideEffect, ToolContext, ToolOutcome, ToolSpec
from ases.kernel.tools.registry import InvocationVerdict, ToolRegistry

_POLICIES_DIR = Path(__file__).parents[2] / "src/orchestrator/ases/policies"


async def _allow_handler(args: object, ctx: object) -> ToolOutcome:
    return ToolOutcome(ok=True)


@pytest.mark.invariant
async def test_an_unregistered_tool_is_refused() -> None:
    """CLAUDE.md section 6: 'Unknown tools/actions are denied.'"""
    registry = ToolRegistry()
    result = await registry.invoke(
        "no.such.tool", {}, ToolContext(cwd=PurePosixPath("/sandbox"), run_id="r")
    )
    assert result.denied is True
    assert result.verdict is InvocationVerdict.DENIED_UNKNOWN_TOOL


@pytest.mark.invariant
async def test_a_write_outside_the_sandbox_is_refused() -> None:
    """CLAUDE.md section 3: "Code changes occur only inside an isolated
    sandbox/workspace." Enforced structurally by the registry, before any
    handler runs - see the confirmed exploit this closes in
    `classification.py`'s `is_safe_relative_path` docstring."""
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="write",
            handler=_allow_handler,
            idempotent=True,
            side_effect=SideEffect.SANDBOX,
            writable_paths=(PurePosixPath("."),),
            path_args=("path",),
        )
    )
    result = await registry.invoke(
        "write",
        {"path": "../outside-the-sandbox.txt"},
        ToolContext(cwd=PurePosixPath("/sandbox"), run_id="r"),
    )
    assert result.denied is True
    assert result.verdict is InvocationVerdict.DENIED_WRITE_OUTSIDE_SANDBOX


@pytest.mark.invariant
def test_an_agent_cannot_grant_itself_a_tool_its_manifest_does_not_list() -> None:
    """CLAUDE.md section 3: "Agents cannot modify their own permissions." An
    actor's own manifest is the only thing consulted, and it is external to
    the actor - there is no field on the request itself that can widen it."""
    manifest = CapabilityManifest(actor="implementer", allowed_tools=frozenset({"fs.read_file"}))
    attempt = Action(
        tool_name="dotnet.build",  # not in allowed_tools - a self-escalation attempt
        side_effect=SideEffect.SANDBOX,
        actor_kind=ActorKind.AGENT,
        requires_approval=False,
    )
    result = enforce_capability(manifest, attempt)
    assert result.decision is PolicyDecision.DENY
    assert result.matched_rule == "capability_manifest"


@pytest.mark.invariant
def test_a_tool_cannot_be_the_actor_requesting_another_tools_execution() -> None:
    """CLAUDE.md section 15: "Side effects -> Tools" - a tool call must
    originate from the kernel or an agent, never from another tool, or the
    invariant that every side effect traces back to an accountable actor
    breaks. Enforced by the autonomy policy pack (`policies/autonomy.yaml`)."""
    engine = PolicyEngine(load_policy_rules([_POLICIES_DIR / "autonomy.yaml"]))

    result = engine.evaluate(
        Action(
            tool_name="fs.write_file",
            side_effect=SideEffect.SANDBOX,
            actor_kind=ActorKind.TOOL,
            requires_approval=False,
        ),
        PolicyContext(run_id="r"),
    )
    assert result.decision is PolicyDecision.DENY


@pytest.mark.invariant
def test_a_policy_pack_cannot_loosen_a_tools_own_approval_requirement() -> None:
    """The corollary of "agents cannot modify their own permissions": a YAML
    pack authored by a human also cannot *loosen* what a tool itself
    declared - `PolicyEngine.evaluate` takes the strictest of the tool's own
    floor and every matching rule, never the loosest."""
    permissive_rule = PolicyRule(
        id="try-to-loosen",
        description="",
        decision=PolicyDecision.ALLOW,
        reason="an attempt to override the tool's own requirement",
        when_tool_name="dangerous.tool",
    )
    engine = PolicyEngine((permissive_rule,))
    result = engine.evaluate(
        Action(
            tool_name="dangerous.tool",
            side_effect=SideEffect.EXTERNAL,
            actor_kind=ActorKind.AGENT,
            requires_approval=True,  # the tool's own declared floor
        ),
        PolicyContext(run_id="r"),
    )
    assert result.decision is PolicyDecision.REQUIRE_APPROVAL
