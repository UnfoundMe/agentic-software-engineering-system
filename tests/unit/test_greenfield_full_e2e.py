"""The full Phase 4 architecture, end to end: every agent in
`workflows/greenfield.yaml` (loaded from the real file, not a hand-rolled
copy), both `kind: tool` nodes, and all three human gates, driven through a
live `Scheduler` from one shared `MockProvider` and one shared `ToolRegistry`
with a faked `dotnet` subprocess runner - the same wiring
`agents.wiring.build_greenfield_executors` would hand a real CLI command,
proving the whole architecture Phase 4 built actually completes a run, not
just its individual pieces in isolation.

Deliberately the happy path (`dotnet.test` succeeds first try, so `repair`
never fires, and the security scan finds nothing). What this test does
*not* prove, stated plainly: that a live LLM produces genuinely compiling
C#, or that the six-tool `dotnet` catalog assembles a linked `.sln` (see
`agents/scaffold.py`'s own disclosed simplification) - it proves the
*orchestration* completes correctly end to end with real tool invocations
(real file writes, a real - faked-runner - subprocess call shape) and real
governance (gates, capability checks) at every step.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from uuid import uuid4

import ases
from ases.agents.migration import _MigrationProposal
from ases.agents.wiring import build_greenfield_executors, register_all_prompts
from ases.context.retriever import ArtifactRef, ContextRetriever
from ases.contracts.artifacts import (
    CodePatch,
    DesignSpec,
    DocsPatch,
    FileChange,
    ReleaseReport,
    RequirementSpec,
    ReviewReport,
    SolutionSkeleton,
    TaskGraph,
    TaskSpec,
    TestSuite,
)
from ases.kernel.gates import ApprovalDecision, BudgetEntryGate, BudgetLimits
from ases.kernel.graph import WorkflowGraph
from ases.kernel.scheduler import Scheduler
from ases.kernel.state import NodeStatus, RunStatus
from ases.kernel.store.jsonl import JsonlEventStore
from ases.kernel.tools.dotnet import build_dotnet_tools
from ases.kernel.tools.fs import READ_FILE, WRITE_FILE
from ases.kernel.tools.migrations import MIGRATIONS_CLASSIFY, build_ef_migration_tools
from ases.kernel.tools.registry import ToolRegistry
from ases.kernel.tools.security import SCAN_FOR_SECRETS
from ases.providers.base import CompletionResult
from ases.providers.mock import MockProvider
from ases.providers.prompts.registry import PromptRegistry
from ases.providers.router import ModelRouter
from tests.unit.fakes import ScriptedApprovals

WORKFLOWS_DIR = Path(ases.__file__).parent / "workflows"
GENEROUS = BudgetLimits(max_tokens=10_000_000, max_usd=1_000.0, max_wallclock_seconds=3600)


class _AlwaysSucceedsRunner:
    """Stands in for every `dotnet` subprocess call - see
    `tests/unit/test_tool_dotnet.py` for the same pattern applied to the
    tool layer alone. This test needs the *wiring* proven, not a real .NET
    toolchain installed on whatever machine runs the suite."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    async def __call__(self, argv: list[str], cwd: Path) -> tuple[int, str, str]:
        self.calls.append(tuple(argv))
        if argv[:4] == ["dotnet", "ef", "migrations", "script"]:
            return 0, "CREATE TABLE short_urls (id uuid PRIMARY KEY);", ""
        return 0, "", ""


def _build_tools(runner: _AlwaysSucceedsRunner) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WRITE_FILE)
    registry.register(READ_FILE)
    registry.register(SCAN_FOR_SECRETS)
    registry.register(MIGRATIONS_CLASSIFY)
    for spec in build_dotnet_tools(runner):
        registry.register(spec)
    for spec in build_ef_migration_tools(runner):
        registry.register(spec)
    return registry


def _script_responses(provider: MockProvider) -> None:
    """One scripted response per prompt_version - `implementer.write@v4` is
    queued twice, once each for `impl_domain` and `impl_api` (see
    `providers/mock.py`'s `respond_to`: matched by prompt_version, safe under
    the two parallel implementation tasks sharing this one provider).

    `implementer`/`tester`/`docs` are not pinned to `@v1`: live runs showed
    `@v1`'s "path relative to the sandbox root" wording is ambiguous with
    `fs.write_file`'s actual `tool_cwd`, and (`implementer` only, hence its
    further bumps to `@v4` while `tester`/`docs` stay at `@v2`) that this
    agent kept regenerating `.csproj` files from scratch and silently
    reverting their target framework, and that two concurrent repair
    positions could silently clobber a shared solution-wide file - see
    `agents/implementer.py`'s `PROMPT_VERSION` comment for all three."""
    req_spec = RequirementSpec(
        summary="A URL shortener: create + redirect + health",
        in_scope=("create short codes", "redirect", "health check"),
        source_text="Build a URL shortener.",
    )
    design = DesignSpec(
        summary="4-project layered ASP.NET Core solution",
        layers=("Domain", "Application", "Infrastructure", "Api"),
        key_decisions=("EF Core for persistence", "Base62 for short codes"),
    )
    skeleton = SolutionSkeleton(
        projects=("UrlShortener.Domain", "UrlShortener.Api"),
        pinned_packages=("Microsoft.EntityFrameworkCore/10.0.0",),
        frozen_interfaces=("public interface IUrlRepository { }",),
    )
    tasks = TaskGraph(
        tasks=(
            TaskSpec(id="t-domain", description="ShortUrl entity and Base62 encoder"),
            TaskSpec(
                id="t-api",
                description="create/redirect controllers",
                depends_on=("t-domain",),
            ),
        )
    )
    domain_patch = CodePatch(
        summary="domain entity and encoder",
        files=(FileChange(path="Domain/ShortUrl.cs", content="public class ShortUrl { }"),),
    )
    api_patch = CodePatch(
        summary="api controllers",
        files=(FileChange(path="Api/UrlController.cs", content="public class UrlController { }"),),
    )
    test_suite = TestSuite(
        summary="unit tests for ShortUrl and UrlController",
        files=(
            FileChange(path="Tests/ShortUrlTests.cs", content="public class ShortUrlTests { }"),
        ),
        coverage_delta=0.15,
    )
    migration_proposal = _MigrationProposal(
        migration_name="AddShortUrlTable", rationale="adds the short_urls table"
    )
    review = ReviewReport(verdict="pass")
    docs_patch = DocsPatch(
        summary="README quickstart", files=(FileChange(path="README.md", content="# Quickstart"),)
    )
    release = ReleaseReport(ready=True, checklist=("build passed", "tests passed", "scan clean"))

    def ok(text_source: object, model_id: str = "m") -> CompletionResult:
        return CompletionResult(
            text=text_source.model_dump_json(),  # type: ignore[attr-defined]
            parsed=text_source,  # type: ignore[arg-type]
            model_id=model_id,
            stop_reason="end_turn",
        )

    provider.respond_to("requirements.analyze@v1", ok(req_spec))
    provider.respond_to("architect.design@v1", ok(design))
    provider.respond_to("scaffold.plan@v2", ok(skeleton))
    provider.respond_to("decompose.plan@v1", ok(tasks))
    provider.respond_to("implementer.write@v4", ok(domain_patch))
    provider.respond_to("implementer.write@v4", ok(api_patch))
    provider.respond_to("migration.plan@v1", ok(migration_proposal))
    provider.respond_to("tester.write@v2", ok(test_suite))
    provider.respond_to("reviewer.review@v1", ok(review))
    provider.respond_to("docs.write@v2", ok(docs_patch))
    provider.respond_to("release.assess@v1", ok(release))


async def test_the_full_greenfield_graph_completes_end_to_end(tmp_path: Path) -> None:
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")

    provider = MockProvider()
    _script_responses(provider)
    prompts = PromptRegistry()
    register_all_prompts(prompts)
    dotnet_runner = _AlwaysSucceedsRunner()
    tools = _build_tools(dotnet_runner)
    tool_cwd = PurePosixPath(tmp_path.as_posix())

    executors = build_greenfield_executors(
        provider=provider,
        router=ModelRouter(),
        prompts=prompts,
        tools=tools,
        tool_cwd=tool_cwd,
        run_input={"requirement_text": "Build a URL shortener."},
    )
    approvals = ScriptedApprovals(
        {
            "gate1": [ApprovalDecision(granted=True, actor="reviewer")],
            "gate2": [ApprovalDecision(granted=True, actor="reviewer")],
            "migration_gate": [ApprovalDecision(granted=True, actor="reviewer")],
            "gate3": [ApprovalDecision(granted=True, actor="reviewer")],
        }
    )
    store = JsonlEventStore(tmp_path / "events")
    scheduler = Scheduler(
        graph, store, executors, entry_gate=BudgetEntryGate(GENEROUS), approvals=approvals
    )

    state = await scheduler.run(uuid4())

    # -- the whole run reached the end, through every gate ------------------
    assert state.status is RunStatus.COMPLETED
    assert state.approvals["gate1"].granted
    assert state.approvals["gate2"].granted
    assert state.approvals["migration_gate"].granted
    assert state.approvals["gate3"].granted

    # -- every agent node actually ran and succeeded -------------------------
    for node_id in (
        "req",
        "arch",
        "scaffold",
        "decompose",
        "impl_domain",
        "impl_api",
        "build_domain",
        "build_api",
        "migration",
        "migration_apply",
        "test_gen",
        "test_run",
        "review",
        "docs_gen",
        "release",
    ):
        assert state.nodes[node_id].status is NodeStatus.SUCCEEDED, node_id

    # sec_scan found nothing (clean generated code) - it still succeeds, just
    # produces no tracked artifact, per agents/wiring.py's `_scan_artifact`.
    assert state.nodes["sec_scan"].status is NodeStatus.SUCCEEDED
    assert state.nodes["sec_scan"].produced == ()

    # none of the four repair positions fired - the happy path never needed
    # them (every build and every dotnet test succeeded first try).
    for repair_node_id in ("repair", "repair_domain", "repair_api"):
        assert (
            repair_node_id not in state.nodes
            or state.nodes[repair_node_id].status is NodeStatus.PENDING
        )

    # -- real tool invocations actually happened, not just LLM calls --------
    assert (tmp_path / "Domain" / "ShortUrl.cs").is_file()
    assert (tmp_path / "Api" / "UrlController.cs").is_file()
    assert (tmp_path / "Tests" / "ShortUrlTests.cs").is_file()
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "# Quickstart"
    dotnet_new_calls = [
        c for c in dotnet_runner.calls if c[:2] == ("dotnet", "new") and c[2] != "sln"
    ]
    assert len(dotnet_new_calls) == 2  # one per SolutionSkeleton.projects entry
    assert ("dotnet", "new", "sln", "-n", "UrlShortener") in dotnet_runner.calls
    assert ("dotnet", "sln", "add", "UrlShortener.Domain") in dotnet_runner.calls
    assert ("dotnet", "sln", "add", "UrlShortener.Api") in dotnet_runner.calls
    assert (
        "dotnet",
        "add",
        "UrlShortener.Api",
        "reference",
        "UrlShortener.Domain",
    ) in dotnet_runner.calls
    dotnet_build_calls = [c for c in dotnet_runner.calls if c[:2] == ("dotnet", "build")]
    assert len(dotnet_build_calls) == 2  # one per implementation task
    dotnet_test_calls = [c for c in dotnet_runner.calls if c[:2] == ("dotnet", "test")]
    assert len(dotnet_test_calls) == 1
    ef_add_calls = [c for c in dotnet_runner.calls if c[:3] == ("dotnet", "ef", "migrations")]
    assert len(ef_add_calls) == 2  # migrations add + migrations script
    ef_update_calls = [c for c in dotnet_runner.calls if c[:3] == ("dotnet", "ef", "database")]
    assert len(ef_update_calls) == 1

    # -- content genuinely flows between agents, not just hashes ------------
    retriever = ContextRetriever(state)

    # -- the migration was generated and classified, never blindly applied --
    migration_hash = state.nodes["migration"].produced[0]
    fetched_migration = retriever.fetch(ArtifactRef(artifact_hash=migration_hash))
    assert fetched_migration.classification == "safe"
    assert "CREATE TABLE" in fetched_migration.sql

    design_hash = state.nodes["arch"].produced[0]
    fetched_design = retriever.fetch(ArtifactRef(artifact_hash=design_hash))
    assert isinstance(fetched_design, DesignSpec)
    assert "layered" in fetched_design.summary

    release_hash = state.nodes["release"].produced[0]
    fetched_release = retriever.fetch(ArtifactRef(artifact_hash=release_hash))
    assert isinstance(fetched_release, ReleaseReport)
    assert fetched_release.ready is True


async def test_the_full_graph_replays_identically_offline(tmp_path: Path) -> None:
    """Phase 1's resumability property, at full Phase 4 scale: export the
    completed run's event log and re-fold it with no scheduler, no LLM, and
    no tool registry involved at all."""
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    provider = MockProvider()
    _script_responses(provider)
    prompts = PromptRegistry()
    register_all_prompts(prompts)
    tools = _build_tools(_AlwaysSucceedsRunner())
    tool_cwd = PurePosixPath(tmp_path.as_posix())

    executors = build_greenfield_executors(
        provider=provider,
        router=ModelRouter(),
        prompts=prompts,
        tools=tools,
        tool_cwd=tool_cwd,
        run_input={"requirement_text": "Build a URL shortener."},
    )
    approvals = ScriptedApprovals(
        {
            k: [ApprovalDecision(granted=True, actor="reviewer")]
            for k in ("gate1", "gate2", "migration_gate", "gate3")
        }
    )
    store = JsonlEventStore(tmp_path / "events")
    run_id = uuid4()
    await Scheduler(
        graph, store, executors, entry_gate=BudgetEntryGate(GENEROUS), approvals=approvals
    ).run(run_id)

    from ases.kernel.state import fold

    events = [e async for e in store.read(run_id)]
    replayed = fold(run_id, events)

    assert replayed.status is RunStatus.COMPLETED
    assert replayed.nodes["release"].status is NodeStatus.SUCCEEDED
    release_hash = replayed.nodes["release"].produced[0]
    assert replayed.artifact_content[release_hash]["ready"] is True


async def test_a_rejected_migration_gate_never_applies_before_a_decision_exists(
    tmp_path: Path,
) -> None:
    """docs/04 section 4.1's governing principle, at the graph level: a human
    who has not yet approved `migration_gate` must never see
    `ef.database_update` run - checked at the instant of rejection, before
    the redo cycle (below) has any chance to eventually approve a later
    attempt."""
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    provider = MockProvider()
    _script_responses(provider)
    prompts = PromptRegistry()
    register_all_prompts(prompts)
    dotnet_runner = _AlwaysSucceedsRunner()
    tools = _build_tools(dotnet_runner)
    tool_cwd = PurePosixPath(tmp_path.as_posix())

    executors = build_greenfield_executors(
        provider=provider,
        router=ModelRouter(),
        prompts=prompts,
        tools=tools,
        tool_cwd=tool_cwd,
        run_input={"requirement_text": "Build a URL shortener."},
    )
    approvals = ScriptedApprovals(
        {
            "gate1": [ApprovalDecision(granted=True, actor="reviewer")],
            "gate2": [ApprovalDecision(granted=True, actor="reviewer")],
            "migration_gate": [
                ApprovalDecision(granted=False, actor="reviewer", reason="needs a rollback plan")
            ],
            "gate3": [ApprovalDecision(granted=True, actor="reviewer")],
        }
    )
    store = JsonlEventStore(tmp_path / "events")
    scheduler = Scheduler(
        graph, store, executors, entry_gate=BudgetEntryGate(GENEROUS), approvals=approvals
    )

    state = await scheduler.run(uuid4())

    # migration_gate's rejection now retries `migration` (see the redo test
    # below) - with only one scripted `migration.plan@v1` response, the retry
    # itself fails (MockProvider has nothing left to serve it), so the run as
    # a whole fails. What matters here is unconditional regardless of that:
    # `ef.database_update` never ran while the only decision on record was a
    # rejection.
    assert state.status is not RunStatus.COMPLETED
    assert state.approvals["migration_gate"].granted is False
    assert state.approvals["migration_gate"].reason == "needs a rollback plan"
    ef_update_calls = [c for c in dotnet_runner.calls if c[:3] == ("dotnet", "ef", "database")]
    assert ef_update_calls == []


async def test_a_rejected_migration_gate_can_be_redone_and_then_approved(tmp_path: Path) -> None:
    """The behavior a live run actually needs: rejecting `migration_gate`
    with a reason re-dispatches `migration` (not a dead end), and approving
    the second attempt lets the run reach completion, applying only the
    approved migration."""
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    provider = MockProvider()
    _script_responses(provider)
    # A second `migration.plan@v1` response for the redo, with a distinct
    # name - `MigrationAgent` asks the model to pick one different from its
    # own prior attempt, since `ef.migrations_add` is not idempotent.
    provider.respond_to(
        "migration.plan@v1",
        CompletionResult(
            text=_MigrationProposal(
                migration_name="AddShortUrlTableV2", rationale="adds a rollback-safe version"
            ).model_dump_json(),
            parsed=_MigrationProposal(
                migration_name="AddShortUrlTableV2", rationale="adds a rollback-safe version"
            ),
            model_id="m",
            stop_reason="end_turn",
        ),
    )
    prompts = PromptRegistry()
    register_all_prompts(prompts)
    dotnet_runner = _AlwaysSucceedsRunner()
    tools = _build_tools(dotnet_runner)
    tool_cwd = PurePosixPath(tmp_path.as_posix())

    executors = build_greenfield_executors(
        provider=provider,
        router=ModelRouter(),
        prompts=prompts,
        tools=tools,
        tool_cwd=tool_cwd,
        run_input={"requirement_text": "Build a URL shortener."},
    )
    approvals = ScriptedApprovals(
        {
            "gate1": [ApprovalDecision(granted=True, actor="reviewer")],
            "gate2": [ApprovalDecision(granted=True, actor="reviewer")],
            "migration_gate": [
                ApprovalDecision(granted=False, actor="reviewer", reason="needs a rollback plan"),
                ApprovalDecision(granted=True, actor="reviewer", reason="looks safe now"),
            ],
            "gate3": [ApprovalDecision(granted=True, actor="reviewer")],
        }
    )
    store = JsonlEventStore(tmp_path / "events")
    scheduler = Scheduler(
        graph, store, executors, entry_gate=BudgetEntryGate(GENEROUS), approvals=approvals
    )

    state = await scheduler.run(uuid4())

    assert state.status is RunStatus.COMPLETED
    assert state.approvals["migration_gate"].granted is True
    assert state.nodes["migration"].attempt == 2

    ef_add_calls = [
        tuple(c) for c in dotnet_runner.calls if c[:4] == ("dotnet", "ef", "migrations", "add")
    ]
    assert ef_add_calls == [
        (
            "dotnet",
            "ef",
            "migrations",
            "add",
            "AddShortUrlTable",
            "--project",
            "UrlShortener.Domain",
            "--startup-project",
            "UrlShortener.Api",
        ),
        (
            "dotnet",
            "ef",
            "migrations",
            "add",
            "AddShortUrlTableV2",
            "--project",
            "UrlShortener.Domain",
            "--startup-project",
            "UrlShortener.Api",
        ),
    ]
    ef_update_calls = [c for c in dotnet_runner.calls if c[:3] == ("dotnet", "ef", "database")]
    assert len(ef_update_calls) == 1  # applied exactly once, only after approval

    retriever = ContextRetriever(state)
    migration_hash = state.nodes["migration"].produced[0]
    final_plan = retriever.fetch(ArtifactRef(artifact_hash=migration_hash))
    assert final_plan.migration_name == "AddShortUrlTableV2"


async def test_gate2_rejected_to_its_cycle_budget_halts_cleanly(tmp_path: Path) -> None:
    """The other half of the same guarantee gate1's clarification cycle
    already had: a human who keeps rejecting must eventually get a clean
    `HALTED`, not an infinite loop - `arch`'s `cycle_budget: 3` bounds it."""
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")
    provider = MockProvider()
    req_spec = RequirementSpec(summary="s", source_text="raw")
    provider.respond_to(
        "requirements.analyze@v1",
        CompletionResult(
            text=req_spec.model_dump_json(), parsed=req_spec, model_id="m", stop_reason="end_turn"
        ),
    )
    design = DesignSpec(summary="d")
    for _ in range(3):
        provider.respond_to(
            "architect.design@v1",
            CompletionResult(
                text=design.model_dump_json(),
                parsed=design,
                model_id="m",
                stop_reason="end_turn",
            ),
        )
    prompts = PromptRegistry()
    register_all_prompts(prompts)
    tools = _build_tools(_AlwaysSucceedsRunner())
    tool_cwd = PurePosixPath(tmp_path.as_posix())

    executors = build_greenfield_executors(
        provider=provider,
        router=ModelRouter(),
        prompts=prompts,
        tools=tools,
        tool_cwd=tool_cwd,
        run_input={"requirement_text": "Build a URL shortener."},
    )
    approvals = ScriptedApprovals(
        {
            "gate1": [ApprovalDecision(granted=True, actor="reviewer")],
            "gate2": [
                ApprovalDecision(granted=False, actor="reviewer", reason="no"),
                ApprovalDecision(granted=False, actor="reviewer", reason="still no"),
                ApprovalDecision(granted=False, actor="reviewer", reason="still no"),
            ],
        }
    )
    store = JsonlEventStore(tmp_path / "events")
    scheduler = Scheduler(
        graph, store, executors, entry_gate=BudgetEntryGate(GENEROUS), approvals=approvals
    )

    state = await scheduler.run(uuid4())

    assert state.status is RunStatus.HALTED
    assert state.halt_reason is not None
    assert "cycle_budget" in state.halt_reason
