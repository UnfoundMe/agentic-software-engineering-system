"""The `docs` agent (docs/03 section 3.3's `DOCS_GEN` node)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

from ases.agents.base import AgentContext
from ases.agents.docs import DocsAgent, register_prompts
from ases.context.retriever import ContextRetriever
from ases.contracts.artifacts import CodePatch, DesignSpec, DocsPatch, FileChange
from ases.kernel.state import ArtifactRecord, NodeState, RunState
from ases.kernel.tools.fs import WRITE_FILE
from ases.kernel.tools.registry import ToolRegistry
from ases.providers.base import CompletionResult
from ases.providers.mock import MockProvider
from ases.providers.prompts.registry import PromptRegistry
from ases.providers.router import ModelRouter


def _record(hash_: str, kind: str, node_id: str) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_hash=hash_, kind=kind, node_id=node_id, produced_at=datetime.now(UTC)
    )


def _state() -> RunState:
    state = RunState(run_id=uuid4())
    design = DesignSpec(summary="d")
    domain = CodePatch(summary="domain entity", files=(FileChange(path="A.cs", content="x"),))
    api = CodePatch(summary="api controller", files=(FileChange(path="B.cs", content="y"),))

    state.artifacts["h-design"] = _record("h-design", "DesignSpec", "arch")
    state.artifact_content["h-design"] = design.model_dump(mode="json")
    state.nodes["arch"] = NodeState(node_id="arch", produced=("h-design",))

    state.artifacts["h-domain"] = _record("h-domain", "CodePatch", "impl_domain")
    state.artifact_content["h-domain"] = domain.model_dump(mode="json")
    state.nodes["impl_domain"] = NodeState(node_id="impl_domain", produced=("h-domain",))

    state.artifacts["h-api"] = _record("h-api", "CodePatch", "impl_api")
    state.artifact_content["h-api"] = api.model_dump(mode="json")
    state.nodes["impl_api"] = NodeState(node_id="impl_api", produced=("h-api",))
    return state


def _ctx(provider: MockProvider, state: RunState, tmp_path: Path) -> AgentContext:
    prompts = PromptRegistry()
    register_prompts(prompts)
    registry = ToolRegistry()
    registry.register(WRITE_FILE)
    return AgentContext(
        run_id="run-1",
        node_id="docs_gen",
        retriever=ContextRetriever(state),
        provider=provider,
        router=ModelRouter(),
        prompts=prompts,
        tools=registry,
        tool_cwd=PurePosixPath(tmp_path.as_posix()),
        capabilities=DocsAgent.capabilities,
    )


def test_build_input_reads_both_implementations(tmp_path: Path) -> None:
    ctx = _ctx(MockProvider(), _state(), tmp_path)
    inp = DocsAgent().build_input(ctx)
    assert len(inp.implementations) == 2


async def test_run_writes_doc_files_for_real(tmp_path: Path) -> None:
    patch = DocsPatch(
        summary="added README", files=(FileChange(path="README.md", content="# Quickstart"),)
    )
    provider = MockProvider()
    provider.respond_with(
        CompletionResult(
            text=patch.model_dump_json(), parsed=patch, model_id="m", stop_reason="end_turn"
        )
    )
    ctx = _ctx(provider, _state(), tmp_path)
    agent = DocsAgent()

    result = await agent.run(ctx, agent.build_input(ctx))

    assert result.artifact == patch
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "# Quickstart"


def test_capability_manifest_grants_exactly_fs_write_file() -> None:
    assert DocsAgent.capabilities.allowed_tools == frozenset({"fs.write_file"})
