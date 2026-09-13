"""The `scaffold` agent (docs/02 section 7.1; docs/03 section 3.3).

The first agent under test that actually reaches a real tool - `dotnet.new`
- through the mechanism `agents.base.AgentContext.invoke_tool` provides.
Following `kernel/tools/test_tool_dotnet.py`'s own convention, the subprocess
runner is faked so no real `dotnet` binary is required; what's under test
here is the *wiring* (capability check, tool_cwd, argument passing), not the
.NET toolchain itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

import pytest

from ases.agents.base import AgentContext
from ases.agents.scaffold import (
    ScaffoldAgent,
    ScaffoldAgentInput,
    ScaffoldToolFailureError,
    register_prompts,
)
from ases.context.retriever import ContextRetriever, NoArtifactFromNodeError
from ases.contracts.artifacts import DesignSpec, SolutionSkeleton
from ases.kernel.state import ArtifactRecord, NodeState, RunState
from ases.kernel.tools.dotnet import build_dotnet_tools
from ases.kernel.tools.registry import ToolRegistry
from ases.providers.base import CompletionResult
from ases.providers.mock import MockProvider
from ases.providers.prompts.registry import PromptRegistry
from ases.providers.router import ModelRouter


class _FakeRunner:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.calls: list[Sequence[str]] = []

    async def __call__(self, argv: Sequence[str], cwd: Path) -> tuple[int, str, str]:
        self.calls.append(argv)
        return self.returncode, "", ""


def _registry(runner: _FakeRunner) -> ToolRegistry:
    registry = ToolRegistry()
    for spec in build_dotnet_tools(runner):
        registry.register(spec)
    return registry


def _state_with_design(design: DesignSpec) -> RunState:
    state = RunState(run_id=uuid4())
    state.artifacts["h-design"] = ArtifactRecord(
        artifact_hash="h-design",
        kind="DesignSpec",
        node_id="arch",
        produced_at=datetime.now(UTC),
    )
    state.artifact_content["h-design"] = design.model_dump(mode="json")
    state.nodes["arch"] = NodeState(node_id="arch", produced=("h-design",))
    return state


def _ctx(provider: MockProvider, state: RunState, registry: ToolRegistry) -> AgentContext:
    prompts = PromptRegistry()
    register_prompts(prompts)
    return AgentContext(
        run_id="run-1",
        node_id="scaffold",
        retriever=ContextRetriever(state),
        provider=provider,
        router=ModelRouter(),
        prompts=prompts,
        tools=registry,
        tool_cwd=PurePosixPath("/sandbox"),
        capabilities=ScaffoldAgent.capabilities,
    )


def test_build_input_reads_the_design_from_the_arch_node() -> None:
    design = DesignSpec(summary="4-layer solution")
    agent = ScaffoldAgent()
    ctx = _ctx(MockProvider(), _state_with_design(design), _registry(_FakeRunner()))

    inp = agent.build_input(ctx)

    assert inp == ScaffoldAgentInput(design=design)


def test_build_input_without_an_approved_design_raises() -> None:
    agent = ScaffoldAgent()
    ctx = _ctx(MockProvider(), RunState(run_id=uuid4()), _registry(_FakeRunner()))
    with pytest.raises(NoArtifactFromNodeError):
        agent.build_input(ctx)


async def test_run_materializes_each_planned_project_via_dotnet_new() -> None:
    design = DesignSpec(summary="d")
    skeleton = SolutionSkeleton(
        projects=("UrlShortener.Domain", "UrlShortener.Api"),
        pinned_packages=("Microsoft.EntityFrameworkCore/10.0.0",),
        frozen_interfaces=("public interface IUrlRepository { }",),
    )
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=skeleton.model_dump_json(),
            parsed=skeleton,
            model_id="m",
            stop_reason="end_turn",
        )
    )
    runner = _FakeRunner()
    ctx = _ctx(provider, _state_with_design(design), _registry(runner))
    agent = ScaffoldAgent()

    result = await agent.run(ctx, agent.build_input(ctx))

    assert result.artifact == skeleton
    # new_sln, then per project: new (template by name suffix) + sln_add +
    # (add_reference from the second project on) + one add_package per
    # pinned package - see the module docstring.
    assert runner.calls == [
        ["dotnet", "new", "sln", "-n", "UrlShortener"],
        [
            "dotnet",
            "new",
            "classlib",
            "-n",
            "UrlShortener.Domain",
            "-o",
            "UrlShortener.Domain",
            "-f",
            "net10.0",
        ],
        ["dotnet", "sln", "add", "UrlShortener.Domain"],
        ["dotnet", "add", "UrlShortener.Domain", "package", "Microsoft.EntityFrameworkCore"],
        [
            "dotnet",
            "new",
            "webapi",
            "-n",
            "UrlShortener.Api",
            "-o",
            "UrlShortener.Api",
            "-f",
            "net10.0",
            "-controllers",
        ],
        ["dotnet", "sln", "add", "UrlShortener.Api"],
        ["dotnet", "add", "UrlShortener.Api", "reference", "UrlShortener.Domain"],
        ["dotnet", "add", "UrlShortener.Api", "package", "Microsoft.EntityFrameworkCore"],
    ]


async def test_run_never_passes_version_to_add_package_even_if_the_model_supplied_one() -> None:
    """Regression test: two live runs pinned a package version with a known
    CVE (`NU1903`) and, separately, a stale `net8.0`-era version into this
    same `net10.0` solution (`NU1603`) - either way `-warnaserror` failed
    `build_api`'s entire `cycle_budget`. No `dotnet.add_package` call may
    ever include `--version`, regardless of what `pinned_packages` says."""
    design = DesignSpec(summary="d")
    skeleton = SolutionSkeleton(
        projects=("UrlShortener.Domain",),
        pinned_packages=("SSH.NET/2023.0.0", "Microsoft.Extensions.Caching.Memory/8.0.0"),
    )
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=skeleton.model_dump_json(), parsed=skeleton, model_id="m", stop_reason="end_turn"
        )
    )
    runner = _FakeRunner()
    ctx = _ctx(provider, _state_with_design(design), _registry(runner))
    agent = ScaffoldAgent()

    await agent.run(ctx, agent.build_input(ctx))

    add_package_calls = [call for call in runner.calls if "package" in call]
    assert add_package_calls == [
        ["dotnet", "add", "UrlShortener.Domain", "package", "SSH.NET"],
        ["dotnet", "add", "UrlShortener.Domain", "package", "Microsoft.Extensions.Caching.Memory"],
    ]
    assert not any("--version" in call for call in runner.calls)


async def test_run_derives_a_generic_solution_name_when_projects_share_no_prefix() -> None:
    design = DesignSpec(summary="d")
    skeleton = SolutionSkeleton(projects=("Foo.Domain", "Bar.Api"))
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=skeleton.model_dump_json(), parsed=skeleton, model_id="m", stop_reason="end_turn"
        )
    )
    runner = _FakeRunner()
    ctx = _ctx(provider, _state_with_design(design), _registry(runner))
    agent = ScaffoldAgent()

    await agent.run(ctx, agent.build_input(ctx))

    assert runner.calls[0] == ["dotnet", "new", "sln", "-n", "Solution"]


class _FailAtCallRunner:
    """Every call succeeds except the one at `fail_at_index` (0-based, in
    call order) - lets a test target exactly one step of the
    new_sln/new/sln_add/add_reference sequence."""

    def __init__(self, fail_at_index: int) -> None:
        self._fail_at_index = fail_at_index
        self.calls: list[Sequence[str]] = []

    async def __call__(self, argv: Sequence[str], cwd: Path) -> tuple[int, str, str]:
        index = len(self.calls)
        self.calls.append(argv)
        if index == self._fail_at_index:
            return 1, "", "boom"
        return 0, "", ""


async def test_run_raises_if_new_sln_fails() -> None:
    design = DesignSpec(summary="d")
    skeleton = SolutionSkeleton(projects=("UrlShortener.Domain",))
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=skeleton.model_dump_json(), parsed=skeleton, model_id="m", stop_reason="end_turn"
        )
    )
    ctx = _ctx(provider, _state_with_design(design), _registry(_FailAtCallRunner(0)))
    agent = ScaffoldAgent()

    with pytest.raises(ScaffoldToolFailureError, match=r"dotnet.new_sln"):
        await agent.run(ctx, agent.build_input(ctx))


async def test_run_raises_if_a_project_fails_to_materialize() -> None:
    design = DesignSpec(summary="d")
    skeleton = SolutionSkeleton(projects=("UrlShortener.Domain",))
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=skeleton.model_dump_json(),
            parsed=skeleton,
            model_id="m",
            stop_reason="end_turn",
        )
    )
    ctx = _ctx(provider, _state_with_design(design), _registry(_FailAtCallRunner(1)))
    agent = ScaffoldAgent()

    with pytest.raises(ScaffoldToolFailureError, match=r"UrlShortener\.Domain"):
        await agent.run(ctx, agent.build_input(ctx))


async def test_run_raises_if_sln_add_fails() -> None:
    design = DesignSpec(summary="d")
    skeleton = SolutionSkeleton(projects=("UrlShortener.Domain",))
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=skeleton.model_dump_json(), parsed=skeleton, model_id="m", stop_reason="end_turn"
        )
    )
    ctx = _ctx(provider, _state_with_design(design), _registry(_FailAtCallRunner(2)))
    agent = ScaffoldAgent()

    with pytest.raises(ScaffoldToolFailureError, match=r"dotnet.sln_add"):
        await agent.run(ctx, agent.build_input(ctx))


async def test_run_raises_if_add_reference_fails() -> None:
    design = DesignSpec(summary="d")
    skeleton = SolutionSkeleton(projects=("UrlShortener.Domain", "UrlShortener.Api"))
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=skeleton.model_dump_json(), parsed=skeleton, model_id="m", stop_reason="end_turn"
        )
    )
    # calls: 0=new_sln 1=new(Domain) 2=sln_add(Domain) 3=new(Api) 4=sln_add(Api) 5=add_reference
    ctx = _ctx(provider, _state_with_design(design), _registry(_FailAtCallRunner(5)))
    agent = ScaffoldAgent()

    with pytest.raises(ScaffoldToolFailureError, match=r"dotnet.add_reference"):
        await agent.run(ctx, agent.build_input(ctx))


async def test_run_with_no_projects_invokes_no_tool() -> None:
    design = DesignSpec(summary="d")
    skeleton = SolutionSkeleton()
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=skeleton.model_dump_json(),
            parsed=skeleton,
            model_id="m",
            stop_reason="end_turn",
        )
    )
    runner = _FakeRunner()
    ctx = _ctx(provider, _state_with_design(design), _registry(runner))
    agent = ScaffoldAgent()

    await agent.run(ctx, agent.build_input(ctx))

    assert runner.calls == []


def test_capability_manifest_grants_exactly_the_solution_linking_tools() -> None:
    assert ScaffoldAgent.capabilities.allowed_tools == frozenset(
        {
            "dotnet.new",
            "dotnet.new_sln",
            "dotnet.sln_add",
            "dotnet.add_reference",
            "dotnet.add_package",
        }
    )


@pytest.mark.parametrize(
    ("project", "expected_template"),
    [
        ("UrlShortener.Api", "webapi"),
        ("UrlShortener.api", "webapi"),  # case-insensitive
        ("UrlShortener.Web", "webapi"),
        ("UrlShortener.WebApi", "webapi"),
        ("UrlShortener.Domain", "classlib"),
        ("UrlShortener.Infrastructure", "classlib"),
        ("Api", "webapi"),  # no dot at all - the whole name is the suffix
    ],
)
def test_template_for_selects_webapi_only_for_host_suffixes(
    project: str, expected_template: str
) -> None:
    assert ScaffoldAgent._template_for(project) == expected_template


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        # The prompt now asks for a bare id only - this is the normal case.
        ("Serilog", "Serilog"),
        # Defensive: a version the model includes anyway is stripped, never
        # installed - see `_package_id`'s own docstring on why a version is
        # never passed to `dotnet.add_package` at all.
        ("Microsoft.EntityFrameworkCore/10.0.0", "Microsoft.EntityFrameworkCore"),
        ("Microsoft.EntityFrameworkCore/8.0.0", "Microsoft.EntityFrameworkCore"),
    ],
)
def test_package_id_strips_any_version_the_model_still_includes(
    spec: str, expected: str
) -> None:
    assert ScaffoldAgent._package_id(spec) == expected


async def test_run_raises_if_add_package_fails() -> None:
    design = DesignSpec(summary="d")
    skeleton = SolutionSkeleton(
        projects=("UrlShortener.Domain",),
        pinned_packages=("Microsoft.EntityFrameworkCore/10.0.0",),
    )
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=skeleton.model_dump_json(), parsed=skeleton, model_id="m", stop_reason="end_turn"
        )
    )
    # calls: 0=new_sln 1=new(Domain) 2=sln_add(Domain) 3=add_package(Domain)
    ctx = _ctx(provider, _state_with_design(design), _registry(_FailAtCallRunner(3)))
    agent = ScaffoldAgent()

    with pytest.raises(ScaffoldToolFailureError, match=r"dotnet.add_package"):
        await agent.run(ctx, agent.build_input(ctx))


async def test_run_installs_no_packages_when_none_are_pinned() -> None:
    design = DesignSpec(summary="d")
    skeleton = SolutionSkeleton(projects=("UrlShortener.Domain",))
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=skeleton.model_dump_json(), parsed=skeleton, model_id="m", stop_reason="end_turn"
        )
    )
    runner = _FakeRunner()
    ctx = _ctx(provider, _state_with_design(design), _registry(runner))
    agent = ScaffoldAgent()

    await agent.run(ctx, agent.build_input(ctx))

    assert not any(c[2] == "package" for c in runner.calls if len(c) > 2)
