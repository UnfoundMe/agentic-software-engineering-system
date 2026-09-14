"""Wires every `workflows/greenfield.yaml` handler name to its real
executor - the one place that must be kept in sync with both the agent
roster and the workflow YAML, used by a live `ases run greenfield` (once the
CLI grows one) and by the full end-to-end test alike.

Handler names here are the graph's, not the agent classes' own `name`
attributes - `workflows/greenfield.yaml` names `dotnet_test`/`security_scan`/
`dotnet_build_domain`/`dotnet_build_api`/`dotnet_build_infrastructure`/
`ef_database_update` for its `kind: tool` nodes, which is why
`build_greenfield_executors` maps those explicitly rather than deriving
every entry from `Agent.name`.
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
from ases.agents.planner import (
    TaskGraphSubgraphProvider,
    implementation_patches,
    task_of,
)
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
from ases.contracts.artifacts import SolutionSkeleton
from ases.kernel.failures import FailureKind
from ases.kernel.graph import NodeSpec
from ases.kernel.scheduler import NodeExecutor
from ases.kernel.tools.registry import ToolRegistry
from ases.providers.base import LLMProvider
from ases.providers.prompts.registry import PromptRegistry
from ases.providers.router import ModelRouter

#: Handler name -> `Agent` instance, exactly as `workflows/greenfield.yaml`
#: names them (`req`/`arch`/... nodes declare `handler: requirements`,
#: `handler: architect`, etc.) - `implementer` covers seven graph positions
#: (`impl_domain`, `impl_api`, `impl_infrastructure`, `repair`,
#: `repair_domain`, `repair_api`, `repair_infrastructure`), differentiated by
#: `ctx.node_id` as `agents/implementer.py` documents.
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

#: Which nodes count as implementation output is a property of the run, not
#: of this module. It used to be the tuple
#: `("impl_domain", "impl_api", "impl_infrastructure")` - the three nodes the
#: static graph happened to declare. `agents/docs.py`, `reviewer.py` and
#: `tester.py` each kept their own copy of that idea and each listed only
#: *two* of the three, so the reviewer, the test generator and the docs agent
#: never saw a single line of infrastructure code in any run this system has
#: performed. `implementation_patches` derives it instead, removing both the
#: drift and the hard-coded architecture.


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


def _scan_args(retriever: ContextRetriever, node: NodeSpec) -> Mapping[str, object]:
    """`security.scan_for_secrets` scans every line of code this run wrote -
    concatenated, since the scanner takes one text blob. A secrets scan that
    covers only the layers someone remembered to list is not a secrets scan."""
    contents: list[str] = []
    for patch in implementation_patches(retriever.state):
        contents.extend(f.content for f in patch.files if f.content is not None)
    return {"content": chr(10).join(contents)}


def _skeleton_from(retriever: ContextRetriever) -> SolutionSkeleton | None:
    try:
        skeleton = retriever.fetch_latest_from("scaffold")
    except (NoArtifactFromNodeError, UnknownArtifactError):
        return None
    return skeleton if isinstance(skeleton, SolutionSkeleton) else None


def _dotnet_build_args(retriever: ContextRetriever, node: NodeSpec) -> Mapping[str, object]:
    """The project one `build:<task>` node compiles.

    One handler for every build node in the run, resolving its project from
    the task the node belongs to (`planner.task_of`) rather than from a
    closure fixed at wiring time. There used to be three of these -
    `_build_domain_args`, `_build_api_args`, `_build_infrastructure_args` -
    each finding its project by matching a suffix (`.domain`, `.api`,
    `.infrastructure`) against `SolutionSkeleton.projects`. That is the
    name-based inference this system is not supposed to route on, and it
    could only ever name the three layers someone had written a builder for.

    Falls back to an unscoped `dotnet build` (the whole solution) when the
    node is not a generated task node or the task graph cannot be read - the
    same conservative default the old builders used with no skeleton."""
    task = task_of(retriever.state, node.id)
    if task is None or not task.component:
        return {}
    return {"project": task.component}


def _ef_database_update_args(retriever: ContextRetriever, node: NodeSpec) -> Mapping[str, object]:
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


def _structure_check_args(retriever: ContextRetriever, node: NodeSpec) -> Mapping[str, object]:
    """The scaffolded project list, in the dependency order the architecture
    declared - which is the only architectural input the structural check
    takes. See `validation/structure.py` on why it is expressed that way
    rather than as named layers."""
    skeleton = _skeleton_from(retriever)
    if skeleton is None:
        return {}
    return {"projects": list(skeleton.projects)}


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


def build_greenfield_subgraph_provider() -> TaskGraphSubgraphProvider:
    """The `SubgraphProvider` `kernel.scheduler.Scheduler` needs to admit the
    decomposer's plan into `workflows/greenfield.yaml`'s generic execution
    stage.

    Separate from `build_greenfield_executors` because it is a different kind
    of thing: the executors say how each node does its work, this says which
    nodes exist. The node ids it is given here are the ones that workflow
    declares (`decompose`, `scaffold`, `impl_start`, `impl_end`) - another
    workflow with a differently-shaped lifecycle constructs its own."""
    return TaskGraphSubgraphProvider(
        source_node_id="decompose",
        skeleton_node_id="scaffold",
        start_node_id="impl_start",
        end_node_id="impl_end",
        implementer_handler="implementer",
        build_handler="dotnet_build",
    )


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
    executors["dotnet_test"] = ToolNodeExecutor(
        "dotnet.test", tools=tools, tool_cwd=tool_cwd, failure_kind=FailureKind.BUILD_FAILURE
    )
    # One `dotnet_build` handler for every build node in the run. The nodes
    # are admitted at runtime from the decomposer's TaskGraph
    # (`agents/planner.py`), so neither their count nor their projects are
    # known here - `_dotnet_build_args` resolves each node's project from the
    # task it belongs to. Replaces three handlers hard-coded to one
    # Domain/Api/Infrastructure architecture.
    executors["dotnet_build"] = ToolNodeExecutor(
        "dotnet.build",
        tools=tools,
        tool_cwd=tool_cwd,
        build_args=_dotnet_build_args,
        failure_kind=FailureKind.BUILD_FAILURE,
    )
    executors["structure_check"] = ToolNodeExecutor(
        "structure.check_solution",
        tools=tools,
        tool_cwd=tool_cwd,
        build_args=_structure_check_args,
        failure_kind=FailureKind.BUILD_FAILURE,
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
