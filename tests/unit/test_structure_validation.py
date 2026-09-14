"""Pre-build structural validation of the scaffolded solution.

The cheap half of "catch it before the compiler does". Live run
`009ea59f-...` spent three builds and two repair attempts before safe-stopping
with a halt reason that named the wrong node; a structural fault that can be
read off the `.csproj` files should cost none of that.

Nothing here knows what a "domain" or an "API" is. The only architectural
input is the order the projects were declared in, which came from the
architecture agent by way of the scaffold.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from uuid import uuid4

from ases.kernel.tools.classification import SideEffect, ToolContext
from ases.kernel.tools.registry import ToolRegistry
from ases.validation.structure import (
    check_reference_graph,
    project_name_of,
    references_in,
)
from ases.validation.structure_tool import CHECK_SOLUTION

LAYERS = ("Shop.Domain", "Shop.Application", "Shop.Infrastructure", "Shop.Api")


def _csproj(*references: str) -> str:
    items = "\n".join(f'    <ProjectReference Include="..\\{r}\\{r}.csproj" />' for r in references)
    return f"""<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><TargetFramework>net10.0</TargetFramework></PropertyGroup>
  <ItemGroup>
{items}
  </ItemGroup>
</Project>
"""


# --- parsing ---------------------------------------------------------------


def test_a_reference_path_resolves_to_a_project_name() -> None:
    assert project_name_of("..\\Shop.Domain\\Shop.Domain.csproj") == "Shop.Domain"
    # A `.csproj` written by `dotnet` on Linux uses the other separator, and
    # neither spelling is wrong.
    assert project_name_of("../Shop.Domain/Shop.Domain.csproj") == "Shop.Domain"


def test_references_are_found_regardless_of_attribute_spelling() -> None:
    text = (
        "<ItemGroup>\n"
        '  <ProjectReference Include="../A/A.csproj" />\n'
        "  <ProjectReference Condition=\"'$(X)'=='1'\" "
        "Include='../B/B.csproj'></ProjectReference>\n"
        "</ItemGroup>"
    )
    assert references_in(text) == ("A", "B")


def test_a_malformed_csproj_yields_no_references_rather_than_raising() -> None:
    """A half-written project file should read as "no references", never as a
    parse exception failing a run for a reason unrelated to the check."""
    assert references_in("<Project><ItemGroup><ProjectRef") == ()


# --- the reference graph ---------------------------------------------------


def test_a_correctly_layered_solution_has_no_findings() -> None:
    report = check_reference_graph(
        projects=LAYERS,
        csproj_by_project={
            "Shop.Domain": _csproj(),
            "Shop.Application": _csproj("Shop.Domain"),
            "Shop.Infrastructure": _csproj("Shop.Application"),
            "Shop.Api": _csproj("Shop.Infrastructure"),
        },
    )

    assert report.ok
    assert report.summary() == "no structural problems found"


def test_a_dependency_pointing_the_wrong_way_is_reported() -> None:
    """CLAUDE.md section 9: Application must not depend on Infrastructure.
    Expressed generically - Application is declared before Infrastructure, so
    referencing it is an inversion whatever the two layers are called."""
    report = check_reference_graph(
        projects=LAYERS,
        csproj_by_project={
            "Shop.Application": _csproj("Shop.Domain", "Shop.Infrastructure"),
        },
    )

    assert not report.ok
    finding = report.findings[0]
    assert finding.rule == "inverted_dependency"
    assert finding.project == "Shop.Application"
    assert "Shop.Infrastructure" in finding.message


def test_the_permitted_direction_is_not_reported() -> None:
    """The mirror of the test above: Infrastructure referencing Application
    is exactly what the layering requires, and must stay silent."""
    report = check_reference_graph(
        projects=LAYERS,
        csproj_by_project={"Shop.Infrastructure": _csproj("Shop.Application", "Shop.Domain")},
    )

    assert report.ok


def test_a_reference_to_a_project_outside_the_solution_is_reported() -> None:
    report = check_reference_graph(
        projects=LAYERS, csproj_by_project={"Shop.Api": _csproj("Shop.Ghost")}
    )

    assert [f.rule for f in report.findings] == ["unknown_reference"]
    assert "Shop.Ghost" in report.findings[0].message


def test_a_reference_cycle_is_reported_as_a_path() -> None:
    """MSBuild rejects a cycle too, but names an arbitrary member of it."""
    report = check_reference_graph(
        projects=("A", "B"),
        csproj_by_project={"A": _csproj("B"), "B": _csproj("A")},
    )

    rules = [f.rule for f in report.findings]
    assert "reference_cycle" in rules
    cycle = next(f for f in report.findings if f.rule == "reference_cycle")
    assert "->" in cycle.message


def test_a_project_whose_csproj_cannot_be_read_is_skipped_not_reported() -> None:
    """This runs against whatever the sandbox actually contains. "I could not
    read that file" is not a structural finding about the architecture - the
    scaffold step reports its own failures."""
    report = check_reference_graph(projects=LAYERS, csproj_by_project={})

    assert report.ok


# --- the registered tool ---------------------------------------------------


def _write_solution(root: Path, layout: dict[str, list[str]]) -> None:
    for project, references in layout.items():
        directory = root / project
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{project}.csproj").write_text(_csproj(*references), encoding="utf-8")


async def test_the_tool_reads_the_sandbox_and_passes_a_sound_solution(tmp_path: Path) -> None:
    _write_solution(
        tmp_path,
        {
            "Shop.Domain": [],
            "Shop.Application": ["Shop.Domain"],
            "Shop.Infrastructure": ["Shop.Application"],
            "Shop.Api": ["Shop.Infrastructure"],
        },
    )
    registry = ToolRegistry()
    registry.register(CHECK_SOLUTION)

    result = await registry.invoke(
        "structure.check_solution",
        {"projects": list(LAYERS)},
        ToolContext(cwd=PurePosixPath(tmp_path.as_posix()), run_id=str(uuid4()), node_id="sc"),
    )

    assert result.ok
    assert result.output["checked"] == 4


async def test_the_tool_fails_the_node_on_an_inverted_dependency(tmp_path: Path) -> None:
    _write_solution(
        tmp_path,
        {
            "Shop.Domain": [],
            # The layering violation CLAUDE.md section 9 forbids, and the one
            # a compiler would only surface indirectly, much later.
            "Shop.Application": ["Shop.Domain", "Shop.Infrastructure"],
            "Shop.Infrastructure": ["Shop.Application"],
            "Shop.Api": ["Shop.Infrastructure"],
        },
    )
    registry = ToolRegistry()
    registry.register(CHECK_SOLUTION)

    result = await registry.invoke(
        "structure.check_solution",
        {"projects": list(LAYERS)},
        ToolContext(cwd=PurePosixPath(tmp_path.as_posix()), run_id=str(uuid4()), node_id="sc"),
    )

    assert not result.ok
    assert result.error is not None
    assert "inverted_dependency" in result.error


async def test_the_tool_is_read_only_by_declaration() -> None:
    """It runs before the builds and must be incapable of changing anything
    it is about to validate."""
    assert CHECK_SOLUTION.side_effect is SideEffect.NONE
    assert CHECK_SOLUTION.writable_paths == ()
    assert CHECK_SOLUTION.idempotent is True
    assert CHECK_SOLUTION.requires_approval is False


async def test_the_tool_says_so_when_there_is_nothing_to_check(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(CHECK_SOLUTION)

    result = await registry.invoke(
        "structure.check_solution",
        {"projects": []},
        ToolContext(cwd=PurePosixPath(tmp_path.as_posix()), run_id=str(uuid4()), node_id="sc"),
    )

    assert result.ok
    assert result.output["checked"] == 0
