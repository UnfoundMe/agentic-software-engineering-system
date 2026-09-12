"""The `ases` command line: database provisioning, plus audit-log export/replay.

Deliberately thin. Every command here is a wrapper over kernel functions
already tested independently (`kernel/store/bootstrap.py`,
`kernel/store/postgres.py`, `kernel/store/jsonl.py`, `kernel/state.py`) - the
CLI's own job is argument parsing and a legible summary, nothing more.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import UUID

import typer
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from rich.console import Console

from ases.config import REPO_ROOT, settings
from ases.kernel.events import Event
from ases.kernel.state import fold
from ases.kernel.store.base import verify_sequence
from ases.kernel.store.bootstrap import bootstrap as run_bootstrap
from ases.kernel.store.bootstrap import reset_workload as run_reset_workload
from ases.kernel.store.postgres import PostgresEventStore

app = typer.Typer(name="ases", help="Agentic Software Engineering System.", no_args_is_help=True)
db_app = typer.Typer(help="Database provisioning and maintenance.", no_args_is_help=True)
app.add_typer(db_app, name="db")

console = Console()
err_console = Console(stderr=True)


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
