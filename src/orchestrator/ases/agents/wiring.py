"""Wires every `workflows/greenfield.yaml` handler name to its real
executor - the one place that must be kept in sync with both the agent
roster and the workflow YAML, used by a live `ases run greenfield` (once the
CLI grows one) and by the full end-to-end test alike.

Handler names here are the graph's, not the agent classes' own `name`
attributes - `workflows/greenfield.yaml` names `dotnet_test`/`security_scan`/
`dotnet_build_domain`/`dotnet_build_api`/`ef_database_update` for its
`kind: tool` nodes, which is why `build_greenfield_executors` maps those
explicitly rather than deriving every entry from `Agent.name`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from ases.agents.architect import ArchitectAgent
from ases.agents.architect import register_prompts as register_architect_prompts
from ases.agents.base import Agent
from ases.agents.decompose import DecomposeAgent
from ases.agents.decompose import register_prompts as register_decompose_prompts
from ases.agents.docs import DocsAgent
from ases.agents.docs import register_prompts as register_docs_prompts
from ases.agents.executor import AgentNodeExecutor
from ases.agents.implementer import ImplementerAgent
from ases.agents.implementer import register_prompts as register_implementer_prompts
from ases.agents.migration import MigrationAgent, ef_targets
from ases.agents.migration import register_prompts as register_migration_prompts
from ases.agents.release import ReleaseAgent
from ases.agents.release import register_prompts as register_release_prompts
from ases.agents.requirements import RequirementsAgent
from ases.agents.requirements import register_prompts as register_requirements_prompts
from ases.agents.reviewer import ReviewerAgent
from ases.agents.reviewer import register_prompts as register_reviewer_prompts
from ases.agents.scaffold import ScaffoldAgent
from ases.agents.scaffold import register_prompts as register_scaffold_prompts
from ases.agents.tester import TesterAgent
from ases.agents.tester import register_prompts as register_tester_prompts
from ases.agents.tool_executor import ToolNodeExecutor
from ases.context.lineage import UnknownArtifactError
from ases.context.retriever import ContextRetriever, NoArtifactFromNodeError
from ases.contracts.artifacts import CodePatch, SolutionSkeleton
from ases.kernel.scheduler import NodeExecutor
from ases.kernel.tools.registry import ToolRegistry
from ases.providers.base import LLMProvider
from ases.providers.prompts.registry import PromptRegistry
from ases.providers.router import ModelRouter

#: Handler name -> `Agent` instance, exactly as `workflows/greenfield.yaml`
#: names them (`req`/`arch`/... nodes declare `handler: requirements`,
#: `handler: architect`, etc.) - `implementer` covers five graph positions
#: (`impl_domain`, `impl_api`, `repair`, `repair_domain`, `repair_api`),
#: differentiated by `ctx.node_id` as `agents/implementer.py` documents.
_AGENTS: Mapping[str, Agent[Any, Any]] = {
    "requirements": RequirementsAgent(),
    "architect": ArchitectAgent(),
    "scaffold": ScaffoldAgent(),
    "decompose": DecomposeAgent(),
    "implementer": ImplementerAgent(),
    "migration": MigrationAgent(),
    "tester": TesterAgent(),
    "reviewer": ReviewerAgent(),
    "docs": DocsAgent(),
    "release": ReleaseAgent(),
}

#: The two `impl_*` nodes `sec_scan` should read - see `agents/decompose.py`'s
#: docstring on why these are fixed graph positions rather than derived from
#: `TaskGraph.tasks`.
_IMPLEMENTATION_NODE_IDS: tuple[str, ...] = ("impl_domain", "impl_api")

#: Mirrors `agents/migration.py`'s `_WEB_HOST_SUFFIXES` - duplicated rather
#: than imported for the same reason that module gives (a three-word
#: frozenset is not worth a shared module over, and the two call sites should
#: not import each other). Picks the project `build_domain`/`repair_domain`
#: target: `agents/implementer.py`'s own `_FOCUS_BY_NODE_ID` names the domain
#: layer project by this same `.Domain` naming convention.
_DOMAIN_SUFFIXES = frozenset({"domain"})


def register_all_prompts(registry: PromptRegistry) -> None:
    """Registers every Phase 4 agent's prompt template into one registry."""
    register_requirements_prompts(registry)
    register_architect_prompts(registry)
    register_scaffold_prompts(registry)
    register_decompose_prompts(registry)
    register_implementer_prompts(registry)
    register_tester_prompts(registry)
    register_reviewer_prompts(registry)
    register_docs_prompts(registry)
    register_migration_prompts(registry)
    register_release_prompts(registry)


def _scan_args(retriever: ContextRetriever) -> Mapping[str, object]:
    """`security.scan_for_secrets` scans whatever the implementation tasks
    actually wrote - concatenated, since the scanner takes one text blob."""
    contents: list[str] = []
    for node_id in _IMPLEMENTATION_NODE_IDS:
        try:
            patch = retriever.fetch_latest_from(node_id)
        except (NoArtifactFromNodeError, UnknownArtifactError):
            continue
        if isinstance(patch, CodePatch):
            contents.extend(f.content for f in patch.files if f.content is not None)
    return {"content": "\n".join(contents)}


def _domain_project(projects: tuple[str, ...]) -> str:
    if not projects:
        return ""
    return next(
        (p for p in projects if p.rsplit(".", 1)[-1].lower() in _DOMAIN_SUFFIXES),
        projects[0],  # no name matched - fall back to the first declared project
    )


def _skeleton_from(retriever: ContextRetriever) -> SolutionSkeleton | None:
    try:
        skeleton = retriever.fetch_latest_from("scaffold")
    except (NoArtifactFromNodeError, UnknownArtifactError):
        return None
    return skeleton if isinstance(skeleton, SolutionSkeleton) else None


def _build_domain_args(retriever: ContextRetriever) -> Mapping[str, object]:
    """`build_domain`'s own project, so it builds only the domain layer
    instead of falling back to a bare `dotnet build` - which resolves the
    *whole* solution and is exactly what made this node and `build_api`
    (`_build_api_args` below) do literally identical, redundant work. Found
    live (`b5da55c3-...`): both nodes failed with byte-for-byte identical
    diagnostics, then diverged unreproducibly on retry once dispatched
    together by the scheduler's genuine `asyncio.gather` concurrency - see
    `kernel/tools/process.py`'s per-cwd lock for the other half of that fix."""
    skeleton = _skeleton_from(retriever)
    if skeleton is None:
        return {}
    return {"project": _domain_project(skeleton.projects)}


def _build_api_args(retriever: ContextRetriever) -> Mapping[str, object]:
    """`build_api`'s own project - the same `_WEB_HOST_SUFFIXES` naming
    convention `ef_targets` already applies, reused via its `startup_project`
    return value rather than re-implemented here."""
    skeleton = _skeleton_from(retriever)
    if skeleton is None:
        return {}
    _, api_project = ef_targets(skeleton.projects)
    return {"project": api_project}


def _ef_database_update_args(retriever: ContextRetriever) -> Mapping[str, object]:
    """`migration_apply`'s `--project`/`--startup-project` - the same
    `SolutionSkeleton.projects` naming heuristic `agents/migration.py`'s
    `ef_targets` already applies for `ef.migrations_add`/`_script`, reused
    here since `MigrationAgent` itself never invokes `ef.database_update`
    (see its module docstring)."""
    skeleton = _skeleton_from(retriever)
    if skeleton is None:
        return {}
    project, startup_project = ef_targets(skeleton.projects)
    return {"project": project, "startup_project": startup_project}


def _scan_artifact(output: Mapping[str, object]) -> Mapping[str, object] | None:
    findings = output.get("findings")
    if not findings:
        return None
    count = len(findings) if isinstance(findings, list) else "1+"
    return {
        "rule": "secrets_detected",
        "severity": "high",
        "message": f"{count} potential secret(s) found",
    }


def build_greenfield_executors(
    *,
    provider: LLMProvider,
    router: ModelRouter,
    prompts: PromptRegistry,
    tools: ToolRegistry,
    tool_cwd: PurePosixPath,
    run_input: dict[str, str],
) -> dict[str, NodeExecutor]:
    """The `executors` mapping `kernel.scheduler.Scheduler` needs to run
    `workflows/greenfield.yaml` for real. One `provider`/`router` shared
    across every agent, matching production use (`ModelRouter` - not a
    separate provider instance per agent - is what selects a model)."""
    executors: dict[str, NodeExecutor] = {
        handler: AgentNodeExecutor(
            agent,
            provider=provider,
            router=router,
            prompts=prompts,
            run_input=run_input,
            tools=tools,
            tool_cwd=tool_cwd,
        )
        for handler, agent in _AGENTS.items()
    }
    executors["dotnet_test"] = ToolNodeExecutor("dotnet.test", tools=tools, tool_cwd=tool_cwd)
    # Two distinct executors, not one shared `dotnet_build` - each node needs
    # its own project scoped in via `build_args` (see `_build_domain_args`'s
    # docstring for why a single shared, unscoped executor was the root cause
    # of both nodes doing identical, redundant whole-solution builds).
    executors["dotnet_build_domain"] = ToolNodeExecutor(
        "dotnet.build", tools=tools, tool_cwd=tool_cwd, build_args=_build_domain_args
    )
    executors["dotnet_build_api"] = ToolNodeExecutor(
        "dotnet.build", tools=tools, tool_cwd=tool_cwd, build_args=_build_api_args
    )
    executors["ef_database_update"] = ToolNodeExecutor(
        "ef.database_update", tools=tools, tool_cwd=tool_cwd, build_args=_ef_database_update_args
    )
    executors["security_scan"] = ToolNodeExecutor(
        "security.scan_for_secrets",
        tools=tools,
        tool_cwd=tool_cwd,
        build_args=_scan_args,
        artifact_kind="PolicyViolation",
        build_artifact=_scan_artifact,
    )
    return executors
