"""The `migration` agent (docs/04 section 4).

Following `test_agent_scaffold.py`'s own convention: the subprocess runner
behind `ef.migrations_add`/`ef.migrations_script` is faked, so no real EF
Core project or `dotnet` binary is required. What's under test is the
*wiring* - that the agent never invents SQL or a classification itself, and
never touches `ef.database_update` (docs/04 section 4.1's governing
principle).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

import pytest

from ases.agents.base import AgentContext
from ases.agents.migration import (
    MigrationAgent,
    MigrationAgentInput,
    MigrationToolFailureError,
    OpaqueMigrationRejectedError,
    _MigrationProposal,
    ef_targets,
    register_prompts,
)
from ases.context.retriever import ContextRetriever, NoArtifactFromNodeError
from ases.contracts.artifacts import CodePatch, FileChange, MigrationPlan, SolutionSkeleton
from ases.kernel.state import ApprovalRecord, ArtifactRecord, NodeState, RunState
from ases.kernel.tools.migrations import MIGRATIONS_CLASSIFY, build_ef_migration_tools
from ases.kernel.tools.registry import ToolRegistry
from ases.providers.base import CompletionRequest, CompletionResult
from ases.providers.mock import MockProvider
from ases.providers.prompts.registry import PromptRegistry
from ases.providers.router import ModelRouter


class _RecordingMockProvider(MockProvider):
    def __init__(self) -> None:
        super().__init__()
        self.rendered_prompts: list[str] = []

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.rendered_prompts.append(request.rendered_prompt)
        return await super().complete(request)


class _FakeRunner:
    def __init__(self, returncode: int = 0, stdout: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.calls: list[Sequence[str]] = []

    async def __call__(self, argv: Sequence[str], cwd: Path) -> tuple[int, str, str]:
        self.calls.append(argv)
        return self.returncode, self.stdout, ""


def _registry(runner: _FakeRunner) -> ToolRegistry:
    registry = ToolRegistry()
    for spec in build_ef_migration_tools(runner):
        registry.register(spec)
    registry.register(MIGRATIONS_CLASSIFY)
    return registry


_SKELETON = SolutionSkeleton(
    projects=("UrlShortener.Domain", "UrlShortener.Infrastructure", "UrlShortener.Api")
)


def _state_with_domain_patch(patch: CodePatch, skeleton: SolutionSkeleton = _SKELETON) -> RunState:
    state = RunState(run_id=uuid4())
    state.artifacts["h-domain"] = ArtifactRecord(
        artifact_hash="h-domain",
        kind="CodePatch",
        node_id="impl:domain-entities",
        produced_at=datetime.now(UTC),
    )
    state.artifact_content["h-domain"] = patch.model_dump(mode="json")
    state.nodes["impl:domain-entities"] = NodeState(
        node_id="impl:domain-entities", produced=("h-domain",)
    )
    state.artifacts["h-skeleton"] = ArtifactRecord(
        artifact_hash="h-skeleton",
        kind="SolutionSkeleton",
        node_id="scaffold",
        produced_at=datetime.now(UTC),
    )
    state.artifact_content["h-skeleton"] = skeleton.model_dump(mode="json")
    state.nodes["scaffold"] = NodeState(node_id="scaffold", produced=("h-skeleton",))
    return state


def _ctx(provider: MockProvider, state: RunState, registry: ToolRegistry) -> AgentContext:
    prompts = PromptRegistry()
    register_prompts(prompts)
    return AgentContext(
        run_id="run-1",
        node_id="migration",
        retriever=ContextRetriever(state),
        provider=provider,
        router=ModelRouter(),
        prompts=prompts,
        tools=registry,
        tool_cwd=PurePosixPath("/sandbox"),
        capabilities=MigrationAgent.capabilities,
    )


_DOMAIN_PATCH = CodePatch(
    summary="domain entity", files=(FileChange(path="Domain/ShortUrl.cs", content="class C {}"),)
)


def test_build_input_reads_every_implementation_patch_the_run_produced() -> None:
    """This agent used to read exactly one node, `impl_domain`. That node no
    longer exists - implementation nodes are admitted at runtime and which of
    them holds the entity definitions is not knowable from a node id. It
    reads all of them instead: a superset of what it had, containing the
    entity types EF Core needs wherever the architecture put them."""
    agent = MigrationAgent()
    ctx = _ctx(MockProvider(), _state_with_domain_patch(_DOMAIN_PATCH), _registry(_FakeRunner()))

    inp = agent.build_input(ctx)

    assert inp == MigrationAgentInput(implementations=(_DOMAIN_PATCH,), skeleton=_SKELETON)


def test_build_input_without_a_scaffolded_solution_raises() -> None:
    agent = MigrationAgent()
    ctx = _ctx(MockProvider(), RunState(run_id=uuid4()), _registry(_FakeRunner()))
    with pytest.raises(NoArtifactFromNodeError):
        agent.build_input(ctx)


def _proposal_response(name: str = "AddShortUrlTable") -> CompletionResult:
    proposal = _MigrationProposal(migration_name=name, rationale="adds the short_urls table")
    return CompletionResult(
        text=proposal.model_dump_json(), parsed=proposal, model_id="m", stop_reason="end_turn"
    )


async def test_run_generates_and_classifies_a_safe_migration() -> None:
    provider = MockProvider()
    provider.respond_with(_proposal_response())
    runner = _FakeRunner(stdout="CREATE TABLE short_urls (id uuid PRIMARY KEY);")
    ctx = _ctx(provider, _state_with_domain_patch(_DOMAIN_PATCH), _registry(runner))
    agent = MigrationAgent()

    result = await agent.run(ctx, agent.build_input(ctx))

    assert result.artifact.migration_name == "AddShortUrlTable"
    assert result.artifact.classification == "safe"
    assert result.artifact.sql == "CREATE TABLE short_urls (id uuid PRIMARY KEY);"
    # _SKELETON has an "Infrastructure"-suffixed project, so ef_targets picks
    # it as --project and the last project ("...Api") as --startup-project.
    assert runner.calls[0] == [
        "dotnet",
        "ef",
        "migrations",
        "add",
        "AddShortUrlTable",
        "--project",
        "UrlShortener.Infrastructure",
        "--startup-project",
        "UrlShortener.Api",
    ]
    assert runner.calls[1] == [
        "dotnet",
        "ef",
        "migrations",
        "script",
        "--idempotent",
        "--project",
        "UrlShortener.Infrastructure",
        "--startup-project",
        "UrlShortener.Api",
    ]


async def test_run_generates_and_classifies_a_destructive_migration() -> None:
    """DESTRUCTIVE still produces a plan - it is gated by human approval
    downstream (`migration_gate`), not denied by this agent."""
    provider = MockProvider()
    provider.respond_with(_proposal_response())
    runner = _FakeRunner(stdout="DROP TABLE legacy_urls;")
    ctx = _ctx(provider, _state_with_domain_patch(_DOMAIN_PATCH), _registry(runner))
    agent = MigrationAgent()

    result = await agent.run(ctx, agent.build_input(ctx))

    assert result.artifact.classification == "destructive"


async def test_run_raises_if_migrations_add_fails() -> None:
    provider = MockProvider()
    provider.respond_with(_proposal_response())
    runner = _FakeRunner(returncode=1)
    ctx = _ctx(provider, _state_with_domain_patch(_DOMAIN_PATCH), _registry(runner))
    agent = MigrationAgent()

    with pytest.raises(MigrationToolFailureError, match=r"ef\.migrations_add"):
        await agent.run(ctx, agent.build_input(ctx))


async def test_run_rejects_an_opaque_migration() -> None:
    provider = MockProvider()
    provider.respond_with(_proposal_response())
    runner = _FakeRunner(stdout="DO $$ BEGIN NULL; END $$;")
    ctx = _ctx(provider, _state_with_domain_patch(_DOMAIN_PATCH), _registry(runner))
    agent = MigrationAgent()

    with pytest.raises(OpaqueMigrationRejectedError):
        await agent.run(ctx, agent.build_input(ctx))


def _rejected_migration_gate_state(previous_plan: MigrationPlan) -> RunState:
    state = _state_with_domain_patch(_DOMAIN_PATCH)
    state.artifacts["h-migration"] = ArtifactRecord(
        artifact_hash="h-migration",
        kind="MigrationPlan",
        node_id="migration",
        produced_at=datetime.now(UTC),
    )
    state.artifact_content["h-migration"] = previous_plan.model_dump(mode="json")
    state.nodes["migration"] = NodeState(node_id="migration", produced=("h-migration",))
    state.approvals["migration_gate"] = ApprovalRecord(
        node_id="migration_gate",
        artifact_hash="h-migration",
        granted=False,
        actor="human:alice",
        decided_at=datetime.now(UTC),
        reason="needs a rollback plan",
    )
    return state


def test_build_input_reads_the_migration_gate_rejection_reason_and_previous_name() -> None:
    previous_plan = MigrationPlan(
        migration_name="AddShortUrlTable",
        summary="s",
        sql="CREATE TABLE t (id int);",
        classification="safe",
    )
    agent = MigrationAgent()
    ctx = _ctx(
        MockProvider(), _rejected_migration_gate_state(previous_plan), _registry(_FakeRunner())
    )

    inp = agent.build_input(ctx)

    assert inp.prior_rejection == "needs a rollback plan"
    assert inp.previous_migration_name == "AddShortUrlTable"


def test_build_input_with_no_rejection_leaves_both_fields_none() -> None:
    agent = MigrationAgent()
    ctx = _ctx(MockProvider(), _state_with_domain_patch(_DOMAIN_PATCH), _registry(_FakeRunner()))

    inp = agent.build_input(ctx)

    assert inp.prior_rejection is None
    assert inp.previous_migration_name is None


async def test_run_feeds_the_rejection_reason_and_previous_name_into_the_prompt() -> None:
    previous_plan = MigrationPlan(
        migration_name="AddShortUrlTable",
        summary="s",
        sql="CREATE TABLE t (id int);",
        classification="safe",
    )
    provider = _RecordingMockProvider()
    provider.respond_with(_proposal_response(name="AddShortUrlTableV2"))
    runner = _FakeRunner(stdout="CREATE TABLE short_urls (id uuid PRIMARY KEY);")
    ctx = _ctx(provider, _rejected_migration_gate_state(previous_plan), _registry(runner))
    agent = MigrationAgent()

    await agent.run(ctx, agent.build_input(ctx))

    assert len(provider.rendered_prompts) == 1
    prompt = provider.rendered_prompts[0]
    assert "needs a rollback plan" in prompt
    assert "AddShortUrlTable" in prompt  # the previous-name warning


def test_capability_manifest_excludes_database_update() -> None:
    """docs/04 section 4.1: "the agent generates a migration, it does not
    apply one" - `ef.database_update` must never appear here."""
    assert MigrationAgent.capabilities.allowed_tools == frozenset(
        {"ef.migrations_add", "ef.migrations_script", "migrations.classify"}
    )
    assert "ef.database_update" not in MigrationAgent.capabilities.allowed_tools


def test_ef_targets_prefers_an_infrastructure_suffixed_project() -> None:
    projects = ("UrlShortener.Domain", "UrlShortener.Infrastructure", "UrlShortener.Api")
    assert ef_targets(projects) == ("UrlShortener.Infrastructure", "UrlShortener.Api")


def test_ef_targets_falls_back_to_the_second_to_last_project() -> None:
    """No project ends in "Infrastructure" - the project just before the
    startup project (last in the declared dependency order) is used."""
    projects = ("UrlShortener.Domain", "UrlShortener.Api")
    assert ef_targets(projects) == ("UrlShortener.Domain", "UrlShortener.Api")


def test_ef_targets_uses_the_single_project_as_both_when_there_is_only_one() -> None:
    assert ef_targets(("UrlShortener.Api",)) == ("UrlShortener.Api", "UrlShortener.Api")


def test_ef_targets_returns_empty_strings_for_no_projects() -> None:
    assert ef_targets(()) == ("", "")


def test_ef_targets_finds_the_startup_project_by_name_even_when_tests_is_last() -> None:
    """Regression test for a real defect two live runs exposed: `Tests` (not
    `Api`) ends up last in `SolutionSkeleton.projects` whenever the model
    lists a test project at all, since a test project's dependency-order
    position is genuinely after everything else it exercises. The old
    `projects[-1]` heuristic would have picked `UrlShortener.Tests` as
    `--startup-project` - a project with no `Program.cs`/DI composition root
    and none of `Api`'s connection-string configuration, which would have
    failed `migration_apply` (no `ON_FAILURE` recovery edge) the first time
    any run actually reached it."""
    projects = (
        "UrlShortener.Domain",
        "UrlShortener.Application",
        "UrlShortener.Infrastructure",
        "UrlShortener.Api",
        "UrlShortener.Tests",
    )
    assert ef_targets(projects) == ("UrlShortener.Infrastructure", "UrlShortener.Api")


def test_ef_targets_falls_back_to_the_last_project_when_no_name_matches_a_web_host() -> None:
    """No project ends in Api/Web/WebApi at all - the old positional
    heuristic is the only thing left to fall back to."""
    projects = ("UrlShortener.Domain", "UrlShortener.Infrastructure", "UrlShortener.Worker")
    assert ef_targets(projects) == ("UrlShortener.Infrastructure", "UrlShortener.Worker")
