"""The `implementer` agent (docs/03 section 3.3; one class, three graph
positions - `impl_domain`, `impl_api`, `repair`)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

import pytest

from ases.agents.base import AgentContext
from ases.agents.implementer import (
    ImplementerAgent,
    ImplementerToolFailureError,
    register_prompts,
)
from ases.context.retriever import ContextRetriever
from ases.contracts.artifacts import (
    CodePatch,
    DesignSpec,
    FileChange,
    FrozenInterface,
    RequirementSpec,
    SolutionSkeleton,
    TaskGraph,
    TaskSpec,
)
from ases.kernel.state import ArtifactRecord, NodeState, RunState
from ases.kernel.tools.fs import READ_FILE, WRITE_FILE
from ases.kernel.tools.registry import ToolRegistry
from ases.providers.base import CompletionResult
from ases.providers.mock import MockProvider
from ases.providers.prompts.registry import PromptRegistry
from ases.providers.router import ModelRouter


def _record(hash_: str, kind: str, node_id: str) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_hash=hash_, kind=kind, node_id=node_id, produced_at=datetime.now(UTC)
    )


def _base_state() -> RunState:
    state = RunState(run_id=uuid4())
    requirement = RequirementSpec(summary="r", source_text="build a thing")
    design = DesignSpec(summary="d")
    skeleton = SolutionSkeleton(
        projects=("A", "A.Api"),
        frozen_interfaces=(
            FrozenInterface(
                signature="public interface IX {}",
                namespace="A",
                project="A",
                type_name="IX",
                file_path="A/IX.cs",
            ),
            FrozenInterface(
                signature="public interface IY {}",
                namespace="A.Api",
                project="A.Api",
                type_name="IY",
                file_path="A.Api/IY.cs",
            ),
        ),
    )
    tasks = TaskGraph(
        tasks=(
            TaskSpec(
                id="t1",
                description="the domain entity",
                component="A",
                produces_contracts=("IX",),
            ),
            TaskSpec(
                id="t2",
                description="the HTTP surface",
                component="A.Api",
                depends_on=("t1",),
                consumes_contracts=("IX",),
            ),
        )
    )

    state.artifacts["h-req"] = _record("h-req", "RequirementSpec", "req")
    state.artifact_content["h-req"] = requirement.model_dump(mode="json")
    state.nodes["req"] = NodeState(node_id="req", produced=("h-req",))

    state.artifacts["h-design"] = _record("h-design", "DesignSpec", "arch")
    state.artifact_content["h-design"] = design.model_dump(mode="json")
    state.nodes["arch"] = NodeState(node_id="arch", produced=("h-design",))

    state.artifacts["h-skel"] = _record("h-skel", "SolutionSkeleton", "scaffold")
    state.artifact_content["h-skel"] = skeleton.model_dump(mode="json")
    state.nodes["scaffold"] = NodeState(node_id="scaffold", produced=("h-skel",))

    state.artifacts["h-tasks"] = _record("h-tasks", "TaskGraph", "decompose")
    state.artifact_content["h-tasks"] = tasks.model_dump(mode="json")
    state.nodes["decompose"] = NodeState(node_id="decompose", produced=("h-tasks",))
    return state


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WRITE_FILE)
    registry.register(READ_FILE)
    return registry


def _ctx(provider: MockProvider, state: RunState, node_id: str, tmp_path: Path) -> AgentContext:
    prompts = PromptRegistry()
    register_prompts(prompts)
    return AgentContext(
        run_id="run-1",
        node_id=node_id,
        retriever=ContextRetriever(state),
        provider=provider,
        router=ModelRouter(),
        prompts=prompts,
        tools=_registry(),
        tool_cwd=PurePosixPath(tmp_path.as_posix()),
        capabilities=ImplementerAgent.capabilities,
    )


def test_build_input_derives_focus_from_the_task_the_node_implements(tmp_path: Path) -> None:
    """The focus text is the decomposer's own task description and component,
    read back from the run's TaskGraph. It used to come from a hard-coded
    table mapping `impl_domain` -> "the domain layer: entities and business
    rules" - this agent having an opinion about what layers a system has,
    which is precisely what stopped the Application layer from ever being
    implemented."""
    ctx = _ctx(MockProvider(), _base_state(), "impl:t1", tmp_path)

    inp = ImplementerAgent().build_input(ctx)

    assert "A" in inp.focus
    assert "the domain entity" in inp.focus
    assert inp.prior_error is None


def test_a_different_task_gets_that_task_s_focus(tmp_path: Path) -> None:
    ctx = _ctx(MockProvider(), _base_state(), "impl:t2", tmp_path)
    inp = ImplementerAgent().build_input(ctx)
    assert "A.Api" in inp.focus
    assert "the HTTP surface" in inp.focus


def test_a_generated_repair_node_reads_its_own_component_s_build_failure(tmp_path: Path) -> None:
    """`repair:A.Api` reads `build:A.Api` - derived from the component the
    node id decodes to (`agents.planner.component_of`), not from a table
    that had to list every repair position in advance. `build`/`repair` are
    keyed by component, not task, so a single-task component like `A.Api`
    (only `t2` targets it) is the simplest case."""
    state = _base_state()
    state.nodes["build:A.Api"] = NodeState(node_id="build:A.Api")
    state.nodes["build:A.Api"].last_error = "CS0246: 'IX' could not be found"
    ctx = _ctx(MockProvider(), state, "repair:A.Api", tmp_path)

    inp = ImplementerAgent().build_input(ctx)

    assert inp.prior_error == "CS0246: 'IX' could not be found"
    assert "A.Api" in inp.focus
    assert "dotnet build" in inp.focus


def test_a_repair_node_does_not_read_a_sibling_component_s_failure(tmp_path: Path) -> None:
    """Each component's repair is scoped to its own build. Handing
    `repair:A.Api` the failure of a project it has no business editing is
    what made `repair_api` unable to act on anything in live run
    `009ea59f-...`."""
    state = _base_state()
    state.nodes["build:A"] = NodeState(node_id="build:A")
    state.nodes["build:A"].last_error = "a failure in a different component"
    ctx = _ctx(MockProvider(), state, "repair:A.Api", tmp_path)

    assert ImplementerAgent().build_input(ctx).prior_error is None


def test_build_input_for_repair_reads_the_prior_test_run_failure(tmp_path: Path) -> None:
    state = _base_state()
    state.nodes["test_run"] = NodeState(node_id="test_run")
    state.nodes["test_run"].last_error = "CS0103: 'Foo' does not exist"
    ctx = _ctx(MockProvider(), state, "repair", tmp_path)

    inp = ImplementerAgent().build_input(ctx)

    assert inp.prior_error == "CS0103: 'Foo' does not exist"
    assert "fixing the failure" in inp.focus


def test_build_input_carries_the_original_requirement_summary(tmp_path: Path) -> None:
    """The model previously had no view of the original requirement at all,
    only the design derived from it - `req` is now fetched unconditionally,
    for every graph position."""
    ctx = _ctx(MockProvider(), _base_state(), "impl:t1", tmp_path)
    inp = ImplementerAgent().build_input(ctx)
    assert inp.requirement_summary == "r"


def test_build_input_scopes_contract_names_to_the_task_s_own_contracts(tmp_path: Path) -> None:
    """A task's `contract_names` is exactly what it declares
    (`produces_contracts` + `consumes_contracts`) - not every frozen
    interface in the solution. `t2` only consumes `IX`; `IY` belongs to no
    task here and must not leak into `t2`'s rendered contract."""
    ctx = _ctx(MockProvider(), _base_state(), "impl:t2", tmp_path)

    inp = ImplementerAgent().build_input(ctx)

    assert inp.contract_names == ("IX",)
    rendered = inp.skeleton.frozen_interface_block(type_names=inp.contract_names)
    assert "IX" in rendered
    assert "IY" not in rendered


def test_build_input_for_a_static_position_renders_every_frozen_interface(
    tmp_path: Path,
) -> None:
    """The static `repair` position has no resolvable task/component scope,
    so `contract_names` stays `None` - meaning "render everything", same as
    before targeted filtering existed."""
    state = _base_state()
    state.nodes["test_run"] = NodeState(node_id="test_run")
    ctx = _ctx(MockProvider(), state, "repair", tmp_path)

    inp = ImplementerAgent().build_input(ctx)

    assert inp.contract_names is None


def test_build_input_collects_a_dependency_s_completed_summary(tmp_path: Path) -> None:
    """`t2` depends on `t1`; once `t1` has produced a `CodePatch`, `t2`'s
    input carries that patch's summary and file paths - never its full file
    content, per the "targeted, not a blind dump" requirement."""
    state = _base_state()
    patch = CodePatch(
        summary="wrote the domain entity",
        files=(FileChange(path="A/ShortUrl.cs", content="public class ShortUrl {}"),),
    )
    state.artifacts["h-impl-t1"] = _record("h-impl-t1", "CodePatch", "impl:t1")
    state.artifact_content["h-impl-t1"] = patch.model_dump(mode="json")
    state.nodes["impl:t1"] = NodeState(node_id="impl:t1", produced=("h-impl-t1",))
    ctx = _ctx(MockProvider(), state, "impl:t2", tmp_path)

    inp = ImplementerAgent().build_input(ctx)

    assert len(inp.dependency_summaries) == 1
    dep = inp.dependency_summaries[0]
    assert dep.task_id == "t1"
    assert dep.summary == "wrote the domain entity"
    assert dep.files == ("A/ShortUrl.cs",)


def test_build_input_omits_a_dependency_that_has_not_produced_anything_yet(
    tmp_path: Path,
) -> None:
    ctx = _ctx(MockProvider(), _base_state(), "impl:t2", tmp_path)
    inp = ImplementerAgent().build_input(ctx)
    assert inp.dependency_summaries == ()


def test_a_repair_node_aggregates_contracts_and_dependencies_across_its_component(
    tmp_path: Path,
) -> None:
    """A `repair:<component>` node's scope spans every task that targets that
    component, not just one (`tasks_of_component`): the focus text, the
    contract union, and the dependency-task ids are all combined across them,
    and a dependency shared by more than one task in the component is only
    summarized once."""
    state = _base_state()
    tasks = TaskGraph(
        tasks=(
            TaskSpec(
                id="t1",
                description="the domain entity",
                component="A",
                produces_contracts=("IX",),
            ),
            TaskSpec(
                id="t2",
                description="the HTTP surface",
                component="A.Api",
                depends_on=("t1",),
                consumes_contracts=("IX",),
            ),
            TaskSpec(
                id="t3",
                description="the health endpoint",
                component="A.Api",
                depends_on=("t1",),
                produces_contracts=("IY",),
            ),
        )
    )
    state.artifact_content["h-tasks"] = tasks.model_dump(mode="json")
    patch = CodePatch(
        summary="wrote the domain entity",
        files=(FileChange(path="A/ShortUrl.cs", content="public class ShortUrl {}"),),
    )
    state.artifacts["h-impl-t1"] = _record("h-impl-t1", "CodePatch", "impl:t1")
    state.artifact_content["h-impl-t1"] = patch.model_dump(mode="json")
    state.nodes["impl:t1"] = NodeState(node_id="impl:t1", produced=("h-impl-t1",))
    state.nodes["build:A.Api"] = NodeState(node_id="build:A.Api")
    state.nodes["build:A.Api"].last_error = "compile error"
    ctx = _ctx(MockProvider(), state, "repair:A.Api", tmp_path)

    inp = ImplementerAgent().build_input(ctx)

    assert "the HTTP surface" in inp.focus
    assert "the health endpoint" in inp.focus
    assert inp.contract_names is not None
    assert set(inp.contract_names) == {"IX", "IY"}
    assert len(inp.dependency_summaries) == 1
    assert inp.dependency_summaries[0].task_id == "t1"


async def test_run_writes_each_file_for_real_into_the_sandbox(tmp_path: Path) -> None:
    patch = CodePatch(
        summary="added entity",
        files=(FileChange(path="ShortUrl.cs", content="public class ShortUrl {}"),),
    )
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=patch.model_dump_json(), parsed=patch, model_id="m", stop_reason="end_turn"
        )
    )
    ctx = _ctx(provider, _base_state(), "impl_domain", tmp_path)
    agent = ImplementerAgent()

    result = await agent.run(ctx, agent.build_input(ctx))

    assert result.artifact == patch
    written = tmp_path / "ShortUrl.cs"
    assert written.is_file()
    assert written.read_text(encoding="utf-8") == "public class ShortUrl {}"


async def test_run_skips_writing_a_file_with_no_content(tmp_path: Path) -> None:
    patch = CodePatch(
        summary="deleted a file",
        files=(FileChange(path="Old.cs", change_kind="delete", content=None),),
    )
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=patch.model_dump_json(), parsed=patch, model_id="m", stop_reason="end_turn"
        )
    )
    ctx = _ctx(provider, _base_state(), "impl_domain", tmp_path)
    agent = ImplementerAgent()

    await agent.run(ctx, agent.build_input(ctx))

    assert not (tmp_path / "Old.cs").exists()


async def test_run_raises_if_a_write_is_refused_by_the_sandbox(tmp_path: Path) -> None:
    """The model proposing an unsafe path (traversal, absolute) must fail
    loudly - the registry's own writable_paths enforcement (Phase 3) already
    refuses it; this proves the implementer does not silently swallow that."""
    patch = CodePatch(
        summary="tries to escape",
        files=(FileChange(path="../outside.cs", content="x"),),
    )
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=patch.model_dump_json(), parsed=patch, model_id="m", stop_reason="end_turn"
        )
    )
    ctx = _ctx(provider, _base_state(), "impl_domain", tmp_path)
    agent = ImplementerAgent()

    with pytest.raises(ImplementerToolFailureError):
        await agent.run(ctx, agent.build_input(ctx))

    assert not (tmp_path.parent / "outside.cs").exists()


def test_capability_manifest_grants_exactly_write_and_read_file() -> None:
    """`fs.read_file` joined `fs.write_file` at `PROMPT_VERSION` 4 - the
    read-before-write half of the shared-file conflict fix needs it."""
    assert ImplementerAgent.capabilities.allowed_tools == frozenset(
        {"fs.write_file", "fs.read_file"}
    )


class _ConcurrentWriterProvider:
    """Scripted `CompletionResult`s like `MockProvider`, but a `complete`
    call numbered in `mutate_on_calls` (1-indexed) also writes to
    `Directory.Build.targets` on disk *before* returning - standing in for a
    different, genuinely concurrent repair task finishing its own write
    while this agent's own completion was in flight. Reusing `MockProvider`
    itself cannot express this: the race is specifically about *when* the
    disk changes relative to `ctx.complete`'s await point, which a plain
    response queue has no way to hook into. Defaults to every call, for a
    competing writer that never stops.
    """

    def __init__(
        self,
        tmp_path: Path,
        results: list[CompletionResult],
        *,
        mutate_on_calls: set[int] | None = None,
    ) -> None:
        self._tmp_path = tmp_path
        self._results = results
        self._mutate_on_calls = mutate_on_calls
        self.call_count = 0

    async def complete(self, request: object) -> CompletionResult:
        result = self._results[min(self.call_count, len(self._results) - 1)]
        self.call_count += 1
        if self._mutate_on_calls is None or self.call_count in self._mutate_on_calls:
            (self._tmp_path / "Directory.Build.targets").write_text(
                f"<Project><!-- external write #{self.call_count} --></Project>", encoding="utf-8"
            )
        return result


def _completion(patch: CodePatch) -> CompletionResult:
    return CompletionResult(
        text=patch.model_dump_json(), parsed=patch, model_id="m", stop_reason="end_turn"
    )


async def test_run_reconciles_once_after_a_shared_file_conflict_then_succeeds(
    tmp_path: Path,
) -> None:
    """Regression test for the run `b5da55c3-...` race: `repair_domain` and
    `repair_api` both regenerated `Directory.Build.targets` from scratch in
    the same scheduling pass, and the second write silently discarded the
    first repair's entire fix. Here, a different task's write lands *between*
    this agent's read and its own write (`_ConcurrentWriterProvider`'s side
    effect) - the exact shape of that race - and one bounded retry must
    re-read the result and merge into it instead of clobbering it."""
    my_patch = CodePatch(
        summary="fix attempt",
        files=(
            FileChange(path="Directory.Build.targets", content="<Project><!-- mine --></Project>"),
        ),
    )
    merged_patch = CodePatch(
        summary="fix attempt, merged with the concurrent write",
        files=(
            FileChange(
                path="Directory.Build.targets",
                content="<Project><!-- external write #1 --><!-- mine --></Project>",
            ),
        ),
    )
    provider = _ConcurrentWriterProvider(
        tmp_path, [_completion(my_patch), _completion(merged_patch)], mutate_on_calls={1}
    )
    ctx = _ctx(provider, _base_state(), "repair_domain", tmp_path)  # type: ignore[arg-type]
    agent = ImplementerAgent()

    result = await agent.run(ctx, agent.build_input(ctx))

    assert result.artifact == merged_patch
    assert provider.call_count == 2  # exactly one reconciliation retry, not a loop
    assert (tmp_path / "Directory.Build.targets").read_text(encoding="utf-8") == (
        "<Project><!-- external write #1 --><!-- mine --></Project>"
    )


async def test_run_raises_after_a_second_shared_file_conflict(tmp_path: Path) -> None:
    """The bounded retry is bounded: if a competing writer keeps changing the
    file on *every* attempt (`_ConcurrentWriterProvider` does this
    unconditionally), the second conflict is unrecoverable and raised exactly
    like any other tool failure - never a silent third attempt."""
    patch = CodePatch(
        summary="fix attempt",
        files=(
            FileChange(path="Directory.Build.targets", content="<Project><!-- mine --></Project>"),
        ),
    )
    provider = _ConcurrentWriterProvider(tmp_path, [_completion(patch), _completion(patch)])
    ctx = _ctx(provider, _base_state(), "repair_domain", tmp_path)  # type: ignore[arg-type]
    agent = ImplementerAgent()

    with pytest.raises(ImplementerToolFailureError):
        await agent.run(ctx, agent.build_input(ctx))

    assert provider.call_count == 2  # exactly the bounded budget, not more
