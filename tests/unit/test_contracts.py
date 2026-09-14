"""Artifact contracts: round-trip cleanly, and stay in sync with the workflow
YAML that names them."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import ases
from ases.contracts import CONTRACTS, Ambiguity, AmbiguityRegister, RequirementSpec
from ases.contracts.artifacts import FrozenInterface, SolutionSkeleton, TaskSpec

WORKFLOWS_DIR = Path(ases.__file__).parent / "workflows"


@pytest.mark.parametrize("kind", sorted(CONTRACTS))
def test_every_contract_has_a_valid_schema(kind: str) -> None:
    """A structural smoke test across every contract: the model
    definition itself must be sound (no bad type hints, no circular refs)
    even before any agent exists to populate one."""
    model = CONTRACTS[kind]
    schema = model.model_json_schema()
    assert schema["title"] == kind
    assert "schema_version" in schema["properties"]


@pytest.mark.parametrize("kind", sorted(CONTRACTS))
def test_every_contract_forbids_unknown_fields(kind: str) -> None:
    model = CONTRACTS[kind]
    assert model.model_config.get("extra") == "forbid"


def test_extra_fields_are_rejected() -> None:
    """L1 validation depends on this: an agent's drift from the schema must
    be visible, not silently accepted."""
    with pytest.raises(Exception, match="extra"):
        RequirementSpec(summary="x", source_text="y", unexpected_field="z")  # type: ignore[call-arg]


def test_ambiguity_register_tracks_resolution() -> None:
    register = AmbiguityRegister(
        ambiguities=(
            Ambiguity(question="daily = calendar day or rolling 24h?", resolution="rolling 24h"),
            Ambiguity(question="sender or receiver limited?"),
        )
    )
    assert not register.all_resolved
    assert register.ambiguities[0].is_resolved
    assert not register.ambiguities[1].is_resolved


def _interface(type_name: str, *, project: str = "P", namespace: str = "N") -> FrozenInterface:
    return FrozenInterface(
        signature=f"public interface {type_name} {{ }}",
        namespace=namespace,
        project=project,
        type_name=type_name,
        file_path=f"{project}/{type_name}.cs",
    )


def test_task_spec_accepts_the_contract_linkage_fields() -> None:
    task = TaskSpec(
        id="t1",
        description="implement the cache adapter",
        component="Infra",
        files=("Infra/RedisCache.cs",),
        produces_contracts=("IShortLinkCache",),
        consumes_contracts=("IShortLinkRepository",),
    )
    assert task.files == ("Infra/RedisCache.cs",)
    assert task.produces_contracts == ("IShortLinkCache",)
    assert task.consumes_contracts == ("IShortLinkRepository",)


def test_solution_skeleton_rejects_a_duplicate_type_name() -> None:
    with pytest.raises(Exception, match="duplicate frozen interface type_name"):
        SolutionSkeleton(
            frozen_interfaces=(
                _interface("IFoo", project="A"),
                _interface("IFoo", project="B"),
            )
        )


def test_solution_skeleton_accepts_distinct_type_names() -> None:
    skeleton = SolutionSkeleton(frozen_interfaces=(_interface("IFoo"), _interface("IBar")))
    assert {i.type_name for i in skeleton.frozen_interfaces} == {"IFoo", "IBar"}


def test_frozen_interface_block_renders_only_the_requested_type_names() -> None:
    skeleton = SolutionSkeleton(frozen_interfaces=(_interface("IFoo"), _interface("IBar")))

    block = skeleton.frozen_interface_block(type_names=("IFoo",))

    assert "IFoo" in block
    assert "IBar" not in block


def test_frozen_interface_block_with_no_matching_type_names_says_none_stated() -> None:
    skeleton = SolutionSkeleton(frozen_interfaces=(_interface("IFoo"),))
    assert skeleton.frozen_interface_block(type_names=("INoSuchType",)) == "(none stated)"


def test_frozen_interface_block_with_no_filter_renders_everything() -> None:
    skeleton = SolutionSkeleton(frozen_interfaces=(_interface("IFoo"), _interface("IBar")))
    block = skeleton.frozen_interface_block()
    assert "IFoo" in block
    assert "IBar" in block


def _produces_kinds_in(path: Path) -> set[str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {n["produces"] for n in raw.get("nodes", []) if n.get("produces")}


def test_every_workflow_produces_kind_has_a_contract() -> None:
    """Cross-check between the workflow graph and the contract registry - the
    same category of guard as tests/invariants/test_layering.py, just for a
    different kind of drift."""
    missing: dict[str, set[str]] = {}
    for workflow_path in WORKFLOWS_DIR.glob("*.yaml"):
        kinds = _produces_kinds_in(workflow_path)
        absent = kinds - set(CONTRACTS)
        if absent:
            missing[workflow_path.name] = absent
    assert not missing, f"workflow(s) name a produces: kind with no contract: {missing}"
