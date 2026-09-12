"""Invariant 1 (structural half): the kernel does not depend on the agent plane.

"The engine owns control flow" is only a real property if the engine *cannot*
consult an agent. A static import check is what makes that enforceable rather
than aspirational - and it is what allows the entire kernel test suite to run
with fake agents and no LLM anywhere in the process.

This test reads source, not imports, so it holds even for modules that are
never imported during a test run.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import ases

PACKAGE_ROOT = Path(ases.__file__).parent
KERNEL_ROOT = PACKAGE_ROOT / "kernel"

#: Packages the kernel must never reach into. Each would let a non-deterministic
#: or outward-facing concern influence control flow.
FORBIDDEN_FOR_KERNEL = ("agents", "providers", "codebase", "validation", "interfaces", "sandbox")

#: Third-party modules that would indicate an LLM call from inside the kernel.
FORBIDDEN_THIRD_PARTY = ("anthropic", "openai", "httpx", "requests")


def _kernel_modules() -> list[Path]:
    return sorted(p for p in KERNEL_ROOT.rglob("*.py") if p.name != "__init__.py")


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_kernel_modules_exist() -> None:
    """Guards the test itself: a passing check over zero files proves nothing."""
    assert _kernel_modules(), "no kernel modules found - the path is probably wrong"


@pytest.mark.invariant
@pytest.mark.parametrize("module", _kernel_modules(), ids=lambda p: p.stem)
def test_kernel_does_not_import_the_agent_plane(module: Path) -> None:
    offenders = {
        name
        for name in _imported_names(module)
        for forbidden in FORBIDDEN_FOR_KERNEL
        if name == f"ases.{forbidden}" or name.startswith(f"ases.{forbidden}.")
    }
    assert not offenders, (
        f"{module.relative_to(PACKAGE_ROOT)} imports {sorted(offenders)}. "
        "The kernel must not depend on the agent plane - see docs/05 section 2."
    )


@pytest.mark.invariant
@pytest.mark.parametrize("module", _kernel_modules(), ids=lambda p: p.stem)
def test_kernel_makes_no_network_calls(module: Path) -> None:
    offenders = {
        name
        for name in _imported_names(module)
        for forbidden in FORBIDDEN_THIRD_PARTY
        if name == forbidden or name.startswith(f"{forbidden}.")
    }
    assert not offenders, (
        f"{module.relative_to(PACKAGE_ROOT)} imports {sorted(offenders)}. "
        "The control plane contains zero LLM calls by invariant."
    )


@pytest.mark.invariant
def test_agent_result_cannot_carry_a_routing_decision() -> None:
    """Invariant 2: an agent returns an artifact, never a next step.

    Skipped until the agent plane exists; written now so the guard lands with
    the first agent rather than after it.
    """
    pytest.importorskip("ases.agents.base", reason="agent plane not implemented yet (Phase 4)")

    from ases.agents.base import AgentResult

    forbidden = {"next_node", "next", "goto", "route", "decision", "status"}
    present = forbidden & set(AgentResult.model_fields)
    assert not present, (
        f"AgentResult exposes {sorted(present)}, which would let an agent steer the graph."
    )
