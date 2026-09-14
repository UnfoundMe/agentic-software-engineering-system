"""The `ases` command line: database provisioning, workflow runs, plus
audit-log export/replay.

Deliberately thin. Every command here is a wrapper over kernel functions
already tested independently (`kernel/store/bootstrap.py`,
`kernel/store/postgres.py`, `kernel/store/jsonl.py`, `kernel/state.py`) - the
CLI's own job is argument parsing and a legible summary, nothing more.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import typer
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from rich.console import Console
from rich.table import Table

import ases
from ases.agents.wiring import (
    build_greenfield_executors,
    build_greenfield_subgraph_provider,
    register_all_prompts,
)
from ases.config import REPO_ROOT, settings
from ases.interfaces.terminal_approval import TerminalApprovalProvider
from ases.kernel.events import Event
from ases.kernel.gates import BudgetEntryGate, BudgetLimits
from ases.kernel.graph import NodeSpec, WorkflowGraph
from ases.kernel.scheduler import NodeExecutionOutcome, NodeExecutor, Scheduler
from ases.kernel.state import InvalidTransitionError, NodeStatus, RunState, RunStatus, fold
from ases.kernel.store.base import verify_sequence
from ases.kernel.store.bootstrap import bootstrap as run_bootstrap
from ases.kernel.store.bootstrap import reset_workload as run_reset_workload
from ases.kernel.store.postgres import PostgresEventStore
from ases.kernel.tools.dotnet import build_dotnet_tools
from ases.kernel.tools.fs import READ_FILE, WRITE_FILE
from ases.kernel.tools.migrations import MIGRATIONS_CLASSIFY, build_ef_migration_tools
from ases.kernel.tools.registry import ToolRegistry
from ases.kernel.tools.security import SCAN_FOR_SECRETS
from ases.providers.factory import MissingApiKeyError, get_provider
from ases.providers.prompts.registry import PromptRegistry
from ases.providers.router import ModelRouter
from ases.sandbox.promotion import PromotionError, promote
from ases.sandbox.workspace import create_workspace
from ases.validation.structure_tool import CHECK_SOLUTION

app = typer.Typer(name="ases", help="Agentic Software Engineering System.", no_args_is_help=True)
db_app = typer.Typer(help="Database provisioning and maintenance.", no_args_is_help=True)
app.add_typer(db_app, name="db")
runs_app = typer.Typer(
    help="Run history: status, sandbox location, rejection reasons.", no_args_is_help=True
)
app.add_typer(runs_app, name="runs")

console = Console()
err_console = Console(stderr=True)

WORKFLOWS_DIR = Path(ases.__file__).parent / "workflows"
#: `.gitignore` names both `.sandbox/` and `sandboxes/`; `sandbox.workspace`'s
#: own default (`.ases-sandboxes`) matches neither, so this command passes
#: an explicit, actually-ignored directory rather than relying on it.
SANDBOXES_DIR = REPO_ROOT / "sandboxes"


def _alembic_config() -> AlembicConfig:
    cfg = AlembicConfig(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "src/orchestrator/ases/migrations"))
    return cfg


@db_app.command("bootstrap")
def db_bootstrap() -> None:
    """Idempotently create roles, schemas, grants and resource limits.

    Safe to run repeatedly - a second run changes nothing and fails nothing
    (docs/04 section 1.7's Phase 0 acceptance criterion).
    """
    asyncio.run(run_bootstrap(settings()))
    console.print("[green]bootstrap complete[/green] - roles, schemas and grants are in place")


@db_app.command("upgrade")
def db_upgrade() -> None:
    """Apply Alembic migrations to the control schema."""
    alembic_command.upgrade(_alembic_config(), "head")
    console.print("[green]migrations applied[/green]")


@db_app.command("reset-workload")
def db_reset_workload(
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
) -> None:
    """Drop and recreate `workload_test`. The audit log is untouched."""
    if not yes:
        typer.confirm(
            "This drops every table in workload_test (agent-generated code, migrations, "
            "data). The control schema's audit log is not affected. Continue?",
            abort=True,
        )
    asyncio.run(run_reset_workload(settings()))
    console.print("[green]workload_test reset[/green] - control schema untouched")


def _build_tool_registry() -> ToolRegistry:
    """Every tool a greenfield run may need, with the real subprocess runner
    (`kernel.tools.process.default_runner`) - the default every
    `build_*_tools` factory already falls back to when no fake is passed."""
    registry = ToolRegistry()
    registry.register(WRITE_FILE)
    registry.register(READ_FILE)
    registry.register(SCAN_FOR_SECRETS)
    registry.register(CHECK_SOLUTION)
    registry.register(MIGRATIONS_CLASSIFY)
    for spec in build_dotnet_tools():
        registry.register(spec)
    for spec in build_ef_migration_tools():
        registry.register(spec)
    return registry


#: Seconds between "still working" heartbeat lines while at least one node
#: is in flight - frequent enough that a real LLM call or `dotnet build`
#: never looks hung, not so frequent it floods the terminal.
_HEARTBEAT_SECONDS = 15.0


class _ProgressExecutor:
    """Wraps a real `NodeExecutor` to print when a node starts and finishes -
    found missing live: between one gate's approval and the next prompt, the
    terminal printed nothing at all while an agent was genuinely working (an
    LLM call, a `dotnet build`, ...) in the background, which looks
    indistinguishable from having hung. `in_flight` is shared across every
    wrapped executor so `_heartbeat` below can report on all of them - nodes
    genuinely run concurrently here (`kernel.scheduler.Scheduler` dispatches
    ready nodes via `asyncio.gather`), so this prints plain lines rather than
    a single animated spinner: two concurrent spinners on one terminal line
    would just fight each other.
    """

    def __init__(self, handler: str, inner: NodeExecutor, in_flight: dict[str, float]) -> None:
        self._handler = handler
        self._inner = inner
        self._in_flight = in_flight

    async def execute(self, node: NodeSpec, state: RunState) -> NodeExecutionOutcome:
        console.print(f"[cyan]-> {node.id}[/cyan] ({self._handler}) started")
        self._in_flight[node.id] = time.monotonic()
        try:
            outcome = await self._inner.execute(node, state)
        finally:
            started_at = self._in_flight.pop(node.id, None)
        elapsed = time.monotonic() - started_at if started_at is not None else 0.0
        if outcome.ok:
            console.print(f"[green]OK[/green] {node.id} succeeded ({elapsed:.1f}s)")
        else:
            console.print(f"[red]FAIL[/red] {node.id} failed ({elapsed:.1f}s): {outcome.error}")
        return outcome


async def _heartbeat(in_flight: dict[str, float]) -> None:
    """Prints a "still working" line every `_HEARTBEAT_SECONDS` for as long
    as anything is in flight - the periodic reassurance a single slow node
    (a big LLM generation, a real `dotnet build`) needs beyond the one-time
    "started" line `_ProgressExecutor` already prints."""
    while True:
        await asyncio.sleep(_HEARTBEAT_SECONDS)
        if not in_flight:
            continue
        now = time.monotonic()
        parts = [f"{node_id} ({now - started:.0f}s)" for node_id, started in in_flight.items()]
        console.print(f"[dim]... still working: {', '.join(parts)}[/dim]")


def _print_run_summary(state: RunState) -> None:
    console.print(f"status:    {state.status}")
    for node_id, node in sorted(state.nodes.items()):
        console.print(f"  {node_id:<16} {node.status.value}")
    console.print(f"usage:     {state.usage.total_tokens} tokens, ${state.usage.usd:.2f}")


def _rejection_reasons(state: RunState) -> list[str]:
    """Every reason a run did not simply succeed - rejected/revoked approvals,
    a safe-stop's halt reason, and failed nodes' last errors - in that order.

    Pulled straight from the folded event log (the source of truth per
    CLAUDE.md section 4), not from anything the sandbox directory itself
    records - a worktree on disk carries no status of its own.
    """
    reasons: list[str] = []
    for approval in state.approvals.values():
        if approval.revoked:
            detail = f" ({approval.revoked_reason})" if approval.revoked_reason else ""
            reasons.append(f"{approval.node_id} revoked{detail}")
        elif not approval.granted:
            detail = f" ({approval.reason})" if approval.reason else ""
            reasons.append(f"{approval.node_id} rejected{detail}")
    if state.halt_reason:
        reasons.append(f"halted: {state.halt_reason}")
    for node_id, node in sorted(state.nodes.items()):
        if node.status is NodeStatus.FAILED and node.last_error:
            reasons.append(f"{node_id} failed: {node.last_error}")
    return reasons


@runs_app.command("list")
def runs_list() -> None:
    """List every run in the audit log with its status, sandbox location (if
    the worktree is still on disk), and why it was rejected/halted/failed, if
    it was.

    The `sandboxes/` directory on its own is just anonymous git worktrees
    named by run_id - this reads the event log (the actual source of truth)
    to answer "what happened to this run" and "where did it end up".
    """
    asyncio.run(_runs_list())


#: Raised by `kernel.state.apply` when an event log does not fold cleanly -
#: a malformed or hand-edited log, never something a normal run produces.
#: One such run must not take down the listing for every other run.
_FOLD_ERRORS: tuple[type[Exception], ...] = (
    InvalidTransitionError,
    ValueError,
    NotImplementedError,
)


async def _fold_safely(
    store: PostgresEventStore, run_id: UUID
) -> tuple[RunState | None, str | None]:
    events = await store.read_all(run_id)
    try:
        return fold(run_id, events), None
    except _FOLD_ERRORS as exc:
        return None, f"{type(exc).__name__}: {exc}"


async def _runs_list() -> None:
    store = PostgresEventStore(settings().app_dsn)
    try:
        run_ids = await store.list_runs()
        entries = [(run_id, *await _fold_safely(store, run_id)) for run_id in run_ids]
    finally:
        await store.close()

    if not entries:
        console.print("[yellow]no runs recorded[/yellow]")
        return

    entries.sort(
        key=lambda e: (
            (e[1].created_at if e[1] is not None and e[1].created_at else None)
            or datetime.min.replace(tzinfo=UTC)
        ),
        reverse=True,
    )

    table = Table()
    table.add_column("run_id")
    table.add_column("status")
    # `overflow="fold"` (wrap, never truncate): a silently ellipsised sandbox
    # path or rejection reason would defeat the point of this command.
    table.add_column("sandbox", overflow="fold")
    table.add_column("reason", overflow="fold")
    for run_id, state, fold_error in entries:
        sandbox_dir = SANDBOXES_DIR / str(run_id)
        sandbox_label = str(sandbox_dir) if sandbox_dir.exists() else "[dim]removed/none[/dim]"
        if state is None:
            table.add_row(
                str(run_id),
                "[red]unreadable[/red]",
                sandbox_label,
                f"[red]event log does not fold: {fold_error}[/red]",
            )
            continue
        reason_label = "; ".join(_rejection_reasons(state))
        table.add_row(str(run_id), state.status.value, sandbox_label, reason_label)
    console.print(table)


@app.command("run")
def run_workflow(
    scenario: str = typer.Argument(
        "greenfield",
        help="Which workflow to run - only 'greenfield' is wired to real executors today.",
    ),
    requirement: str = typer.Option(
        ..., "--requirement", "-r", help="The requirement text to build from."
    ),
    ref: str = typer.Option(
        "HEAD", "--ref", help="Git ref the sandbox worktree is checked out from."
    ),
) -> None:
    """Run `scenario` for real: a live LLM (per ASES_LLM_MODE in .env), a
    real `dotnet`/`git` toolchain, and this terminal as every human approval
    gate.

    Prerequisites this command does not set up for you:
    - `ASES_LLM_MODE=live` (or `record`) and a real `ANTHROPIC_API_KEY` in
      `.env` - the default `replay` mode requires pre-recorded cassettes,
      which will not exist on a first run.
    - `ases db bootstrap` and `ases db upgrade` already run once (the event
      log lives in Postgres).

    Never auto-promotes a rejected or incomplete run. On a `COMPLETED` run
    with `gate3` granted, generated content is copied from the sandbox into
    this repository's tracked `src/url-shortener/` as an uncommitted,
    reviewable `git diff` - nothing is committed or pushed automatically.
    """
    if scenario != "greenfield":
        err_console.print(
            f"[red]no wiring exists for scenario {scenario!r}[/red] - only 'greenfield' is "
            "wired to real executors today (see agents/wiring.py)"
        )
        raise typer.Exit(code=1)
    asyncio.run(_run_greenfield(requirement=requirement, ref=ref))


async def _run_greenfield(*, requirement: str, ref: str) -> None:
    cfg = settings()
    graph = WorkflowGraph.from_yaml(WORKFLOWS_DIR / "greenfield.yaml")

    try:
        provider = get_provider(cfg)
    except MissingApiKeyError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    prompts = PromptRegistry()
    register_all_prompts(prompts)

    run_id = uuid4()
    console.print(f"run_id:    {run_id}")
    console.print(f"llm mode:  {cfg.ases_llm_mode}")

    workspace = await create_workspace(REPO_ROOT, str(run_id), worktrees_dir=SANDBOXES_DIR, ref=ref)
    console.print(f"sandbox:   {workspace.root}")

    tools = _build_tool_registry()
    tool_cwd = workspace.tool_cwd / "src/url-shortener"

    executors = build_greenfield_executors(
        provider=provider,
        router=ModelRouter(),
        prompts=prompts,
        tools=tools,
        tool_cwd=tool_cwd,
        run_input={"requirement_text": requirement},
    )
    in_flight: dict[str, float] = {}
    executors = {
        handler: _ProgressExecutor(handler, executor, in_flight)
        for handler, executor in executors.items()
    }
    entry_gate = BudgetEntryGate(
        BudgetLimits(
            max_tokens=cfg.ases_max_tokens_per_run,
            max_usd=cfg.ases_max_usd_per_run,
            max_wallclock_seconds=cfg.ases_max_wallclock_seconds,
        )
    )
    store = PostgresEventStore(cfg.app_dsn)
    scheduler = Scheduler(
        graph,
        store,
        executors,
        entry_gate=entry_gate,
        approvals=TerminalApprovalProvider(),
        # The implementation nodes this run executes are not in the YAML:
        # they are admitted here, from the decomposer's TaskGraph, once
        # `decompose` succeeds. See `agents/planner.py`.
        subgraphs=build_greenfield_subgraph_provider(),
    )

    heartbeat_task = asyncio.create_task(_heartbeat(in_flight))
    try:
        try:
            state = await scheduler.run(run_id)
        finally:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
            await store.close()
    except Exception as exc:  # deliberately broad: report the real cause, never swallow it
        err_console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
        err_console.print(f"Sandbox left in place for inspection: {workspace.root}")
        raise typer.Exit(code=1) from exc

    console.print()
    console.print("=== Run summary ===")
    _print_run_summary(state)

    gate3 = state.approvals.get("gate3")
    if state.status is RunStatus.COMPLETED and gate3 is not None and gate3.granted:
        try:
            written = promote(workspace, REPO_ROOT)
        except PromotionError as exc:
            err_console.print(f"[yellow]not promoted: {exc}[/yellow]")
        else:
            console.print()
            console.print(
                f"[green]promoted {len(written)} file(s)[/green] into src/url-shortener/:"
            )
            for path in written:
                console.print(f"  {path}")
            console.print(
                "Nothing was committed - review with `git status`/`git diff` and commit "
                "yourself when satisfied."
            )
    else:
        console.print()
        console.print(
            "[yellow]not promoted[/yellow] - the run did not complete with gate3 granted. "
            f"Generated content, if any, remains only in the sandbox: {workspace.root}"
        )


_OUT_OPTION = typer.Option(None, "--out", help="Output path (default: runs/<run_id>/export.jsonl).")
_RUN_ID_ARGUMENT = typer.Argument(..., help="Run to export.")
_FILE_ARGUMENT = typer.Argument(..., exists=True, help="A file written by `ases export`.")


@app.command("export")
def export_run(
    run_id: UUID = _RUN_ID_ARGUMENT,
    out: Path | None = _OUT_OPTION,
) -> None:
    """Export one run's event log to a portable JSONL file.

    Each line is one already-sealed event, written verbatim - this is not a
    re-derivation, it is the exact audit record as Postgres holds it. Pair
    with `ases replay` to fold it back with no database involved at all,
    which is the concrete demonstration that state really is `fold(events)`
    and nothing else.
    """
    destination = out or settings().run_dir(str(run_id)) / "export.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)

    async def _export() -> int:
        store = PostgresEventStore(settings().app_dsn)
        try:
            events = await store.read_all(run_id)
        finally:
            await store.close()
        if not events:
            err_console.print(f"[red]no events found for run {run_id}[/red]")
            raise typer.Exit(code=1)
        with destination.open("w", encoding="utf-8", newline="\n") as fh:
            for event in events:
                fh.write(event.model_dump_json() + "\n")
        return len(events)

    count = asyncio.run(_export())
    console.print(f"[green]exported {count} events[/green] -> {destination}")


@app.command("replay")
def replay_export(file: Path = _FILE_ARGUMENT) -> None:
    """Fold an exported JSONL file with no database involved.

    Verifies the hash chain first - a corrupted or tampered export is
    reported, not silently folded - then prints the same summary a live
    run's state would show.
    """
    events: list[Event] = []
    with file.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(Event.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                err_console.print(f"[red]{file}:{lineno} is not a valid event: {exc}[/red]")
                raise typer.Exit(code=1) from exc

    if not events:
        err_console.print(f"[red]{file} contains no events[/red]")
        raise typer.Exit(code=1)

    run_id = events[0].run_id
    verification = verify_sequence(run_id, events)
    if not verification.ok:
        err_console.print(f"[red]chain verification FAILED for run {run_id}[/red]")
        for problem in verification.problems:
            err_console.print(f"  seq {problem.seq}: {problem.reason}")
        raise typer.Exit(code=1)

    state = fold(run_id, events)
    console.print(f"[green]chain verified[/green] ({verification.events_checked} events)")
    console.print(f"run_id:    {run_id}")
    console.print(f"workflow:  {state.workflow}")
    console.print(f"status:    {state.status}")
    console.print(f"nodes:     {len(state.nodes)}")
    console.print(f"artifacts: {len(state.artifacts)}")
    console.print(f"approvals: {len(state.approvals)}")
    console.print(f"usage:     {state.usage.total_tokens} tokens, ${state.usage.usd:.2f}")


if __name__ == "__main__":
    app()
