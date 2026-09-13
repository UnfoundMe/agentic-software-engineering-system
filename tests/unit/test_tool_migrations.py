"""`kernel.tools.migrations` (docs/04 section 4: generate and classify, never
blindly apply).

Following `test_tool_dotnet.py`'s own convention, every EF-tool test injects
a fake subprocess runner - no real `dotnet` binary or EF Core project is
required. The classifier tests exercise `classify_migration` directly against
docs/04 section 4.3's table.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from ases.kernel.tools.classification import SideEffect, ToolContext
from ases.kernel.tools.migrations import (
    MIGRATIONS_CLASSIFY,
    MigrationClass,
    build_ef_migration_tools,
    classify_migration,
    classify_statement,
)


def _ctx(cwd: str = "/sandbox") -> ToolContext:
    return ToolContext(cwd=PurePosixPath(cwd), run_id="run-1")


class _FakeRunner:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.calls: list[tuple[Sequence[str], Path]] = []

    async def __call__(self, argv: Sequence[str], cwd: Path) -> tuple[int, str, str]:
        self.calls.append((argv, cwd))
        return self.returncode, self.stdout, self.stderr


def _tool(name: str, runner: _FakeRunner):  # type: ignore[no-untyped-def]
    tools = build_ef_migration_tools(runner)
    return next(t for t in tools if t.name == name)


# -- classify_statement / classify_migration ------------------------------


def test_create_table_is_safe() -> None:
    statement = "CREATE TABLE short_urls (id uuid PRIMARY KEY)"
    assert classify_statement(statement) is MigrationClass.SAFE


def test_nullable_add_column_is_safe() -> None:
    assert classify_statement("ALTER TABLE t ADD COLUMN note text") is MigrationClass.SAFE


def test_concurrent_index_is_safe() -> None:
    assert classify_statement("CREATE INDEX CONCURRENTLY ix_t_x ON t (x)") is MigrationClass.SAFE


def test_not_valid_constraint_is_safe() -> None:
    assert (
        classify_statement("ALTER TABLE t ADD CONSTRAINT ck CHECK (x > 0) NOT VALID")
        is MigrationClass.SAFE
    )


def test_create_sequence_is_safe() -> None:
    assert classify_statement("CREATE SEQUENCE t_seq") is MigrationClass.SAFE


def test_not_null_add_column_is_risky_not_safe() -> None:
    assert (
        classify_statement("ALTER TABLE t ADD COLUMN n int NOT NULL DEFAULT 0")
        is MigrationClass.RISKY
    )


def test_non_concurrent_create_index_is_risky() -> None:
    assert classify_statement("CREATE INDEX ix_t_x ON t (x)") is MigrationClass.RISKY


def test_alter_column_type_is_risky() -> None:
    assert classify_statement("ALTER TABLE t ALTER COLUMN x TYPE bigint") is MigrationClass.RISKY


def test_validate_constraint_is_risky() -> None:
    assert classify_statement("ALTER TABLE t VALIDATE CONSTRAINT ck") is MigrationClass.RISKY


def test_drop_table_is_destructive() -> None:
    assert classify_statement("DROP TABLE short_urls") is MigrationClass.DESTRUCTIVE


def test_drop_column_is_destructive() -> None:
    assert classify_statement("ALTER TABLE t DROP COLUMN x") is MigrationClass.DESTRUCTIVE


def test_truncate_is_destructive() -> None:
    assert classify_statement("TRUNCATE t") is MigrationClass.DESTRUCTIVE


def test_delete_from_is_destructive() -> None:
    assert classify_statement("DELETE FROM t WHERE 1=1") is MigrationClass.DESTRUCTIVE


def test_an_unrecognized_statement_shape_is_opaque_not_safe() -> None:
    """docs/04 section 4.3's OPAQUE, reinterpreted as "unrecognized -> deny
    by default" (see the module docstring) - a statement this classifier
    cannot place in any known shape must never default to SAFE."""
    assert classify_statement("DO $$ BEGIN NULL; END $$;") is MigrationClass.OPAQUE


def test_destructive_wins_over_safe_in_the_same_migration() -> None:
    sql = "CREATE TABLE t (id int); DROP TABLE other;"
    assert classify_migration(sql) is MigrationClass.DESTRUCTIVE


def test_empty_migration_is_safe() -> None:
    assert classify_migration("   ;  ;  ") is MigrationClass.SAFE


def test_create_unique_index_is_risky_not_opaque() -> None:
    """`CREATE UNIQUE INDEX` - not `CREATE INDEX` - is exactly the shape EF
    Core emits for any unique constraint (e.g. a `ShortCode` column). A plain
    `CREATE\\s+INDEX` pattern never matches it, since `UNIQUE` sits between
    the two keywords - this fell through to OPAQUE before, which has "no
    approval path at all" (`agents/migration.py`'s
    `OpaqueMigrationRejectedError`)."""
    assert (
        classify_statement('CREATE UNIQUE INDEX "IX_ShortUrls_ShortCode" ON "ShortUrls" ("Code")')
        is MigrationClass.RISKY
    )


def test_concurrent_unique_index_is_safe() -> None:
    assert (
        classify_statement("CREATE UNIQUE INDEX CONCURRENTLY ix_t_x ON t (x)")
        is MigrationClass.SAFE
    )


def test_start_transaction_is_safe() -> None:
    """EF Core's own fixed wrapper, present in every `dotnet ef migrations
    script` output - not a schema change, so it must never be the statement
    that drags an otherwise-safe migration to OPAQUE."""
    assert classify_statement("START TRANSACTION") is MigrationClass.SAFE


def test_begin_transaction_is_safe() -> None:
    assert classify_statement("BEGIN TRANSACTION") is MigrationClass.SAFE


def test_commit_is_safe() -> None:
    assert classify_statement("COMMIT") is MigrationClass.SAFE


def test_ef_migrations_history_bookkeeping_insert_is_safe() -> None:
    assert (
        classify_statement(
            'INSERT INTO "__EFMigrationsHistory" ("MigrationId", "ProductVersion")\n'
            "VALUES ('20260913000000_AddShortUrlTable', '10.0.0')"
        )
        is MigrationClass.SAFE
    )


def test_a_realistic_full_ef_core_migration_script_is_safe_not_opaque() -> None:
    """The exact shape `dotnet ef migrations script` actually produces for a
    first migration adding one table with a unique index - wrapped in a
    transaction, ending with EF's own history bookkeeping row. Every one of
    docs/04's example migrations would have been misclassified OPAQUE by the
    wrapper alone, regardless of what the migration itself changed, before
    the SAFE boilerplate patterns and the UNIQUE INDEX fix above."""
    sql = """\
START TRANSACTION;

CREATE TABLE "ShortUrls" (
    "Id" uuid NOT NULL,
    "ShortCode" text NOT NULL,
    "LongUrl" text NOT NULL,
    "CreatedAtUtc" timestamp with time zone NOT NULL,
    CONSTRAINT "PK_ShortUrls" PRIMARY KEY ("Id")
);

CREATE UNIQUE INDEX "IX_ShortUrls_ShortCode" ON "ShortUrls" ("ShortCode");

INSERT INTO "__EFMigrationsHistory" ("MigrationId", "ProductVersion")
VALUES ('20260913000000_AddShortUrlTable', '10.0.0');

COMMIT;
"""
    assert classify_migration(sql) is MigrationClass.RISKY


# -- migrations.classify tool -----------------------------------------------


async def test_classify_tool_reports_ok_for_a_safe_migration() -> None:
    result = await MIGRATIONS_CLASSIFY.handler({"sql": "CREATE TABLE t (id int);"}, _ctx())
    assert result.ok is True
    assert result.output["classification"] == "safe"


async def test_classify_tool_reports_ok_for_a_destructive_migration() -> None:
    """DESTRUCTIVE still has a path forward (explicit approval) - only OPAQUE
    is denied outright."""
    result = await MIGRATIONS_CLASSIFY.handler({"sql": "DROP TABLE t;"}, _ctx())
    assert result.ok is True
    assert result.output["classification"] == "destructive"


async def test_classify_tool_denies_an_opaque_migration() -> None:
    result = await MIGRATIONS_CLASSIFY.handler({"sql": "DO $$ BEGIN NULL; END $$;"}, _ctx())
    assert result.ok is False
    assert result.output["classification"] == "opaque"
    assert "denied by default" in (result.error or "")


def test_classify_tool_is_side_effect_free_and_idempotent() -> None:
    assert MIGRATIONS_CLASSIFY.side_effect is SideEffect.NONE
    assert MIGRATIONS_CLASSIFY.idempotent is True


# -- build_ef_migration_tools ------------------------------------------------


def test_exactly_the_three_named_ef_subcommands_are_registered() -> None:
    names = {t.name for t in build_ef_migration_tools()}
    assert names == {"ef.migrations_add", "ef.migrations_script", "ef.database_update"}


async def test_ef_migrations_add_invokes_the_correct_argv() -> None:
    runner = _FakeRunner()
    tool = _tool("ef.migrations_add", runner)
    result = await tool.handler({"name": "AddShortUrlTable"}, _ctx())
    assert result.ok is True
    argv, _ = runner.calls[0]
    assert argv == ["dotnet", "ef", "migrations", "add", "AddShortUrlTable"]


async def test_ef_migrations_script_is_idempotent_and_prints_to_stdout() -> None:
    runner = _FakeRunner(stdout="CREATE TABLE t (id int);")
    tool = _tool("ef.migrations_script", runner)
    result = await tool.handler({}, _ctx())
    argv, _ = runner.calls[0]
    assert argv == ["dotnet", "ef", "migrations", "script", "--idempotent"]
    assert result.output["stdout"] == "CREATE TABLE t (id int);"


async def test_ef_database_update_invokes_the_correct_argv() -> None:
    runner = _FakeRunner()
    tool = _tool("ef.database_update", runner)
    await tool.handler({}, _ctx())
    argv, _ = runner.calls[0]
    assert argv == ["dotnet", "ef", "database", "update"]


def test_only_ef_database_update_requires_approval() -> None:
    tools = {t.name: t for t in build_ef_migration_tools()}
    assert tools["ef.database_update"].requires_approval is True
    assert tools["ef.migrations_add"].requires_approval is False
    assert tools["ef.migrations_script"].requires_approval is False


def test_only_ef_database_update_has_external_side_effect() -> None:
    tools = {t.name: t for t in build_ef_migration_tools()}
    assert tools["ef.database_update"].side_effect is SideEffect.EXTERNAL
    assert tools["ef.migrations_add"].side_effect is SideEffect.SANDBOX
    assert tools["ef.migrations_script"].side_effect is SideEffect.NONE


async def test_ef_migrations_add_appends_project_and_startup_project_when_given() -> None:
    runner = _FakeRunner()
    tool = _tool("ef.migrations_add", runner)
    await tool.handler(
        {
            "name": "AddShortUrlTable",
            "project": "UrlShortener.Infrastructure",
            "startup_project": "UrlShortener.Api",
        },
        _ctx(),
    )
    argv, _ = runner.calls[0]
    assert argv == [
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


async def test_ef_migrations_script_appends_project_and_startup_project_when_given() -> None:
    runner = _FakeRunner()
    tool = _tool("ef.migrations_script", runner)
    await tool.handler(
        {"project": "UrlShortener.Infrastructure", "startup_project": "UrlShortener.Api"}, _ctx()
    )
    argv, _ = runner.calls[0]
    assert argv == [
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


async def test_ef_database_update_appends_project_and_startup_project_when_given() -> None:
    runner = _FakeRunner()
    tool = _tool("ef.database_update", runner)
    await tool.handler(
        {"project": "UrlShortener.Infrastructure", "startup_project": "UrlShortener.Api"}, _ctx()
    )
    argv, _ = runner.calls[0]
    assert argv == [
        "dotnet",
        "ef",
        "database",
        "update",
        "--project",
        "UrlShortener.Infrastructure",
        "--startup-project",
        "UrlShortener.Api",
    ]


async def test_ef_targeting_args_are_omitted_when_not_given() -> None:
    """Backward compatible with a caller (or an existing test) that never
    passes them - `dotnet ef`'s own single-project auto-discovery applies."""
    runner = _FakeRunner()
    tool = _tool("ef.migrations_add", runner)
    await tool.handler({"name": "X"}, _ctx())
    argv, _ = runner.calls[0]
    assert "--project" not in argv
    assert "--startup-project" not in argv


async def test_ef_targeting_args_support_project_only() -> None:
    runner = _FakeRunner()
    tool = _tool("ef.migrations_add", runner)
    await tool.handler({"name": "X", "project": "UrlShortener.Infrastructure"}, _ctx())
    argv, _ = runner.calls[0]
    assert argv == [
        "dotnet",
        "ef",
        "migrations",
        "add",
        "X",
        "--project",
        "UrlShortener.Infrastructure",
    ]


def test_ef_database_update_and_migrations_add_declare_a_compensator() -> None:
    tools = {t.name: t for t in build_ef_migration_tools()}
    assert tools["ef.database_update"].compensator is not None
    assert tools["ef.migrations_add"].compensator is not None
    assert tools["ef.migrations_script"].compensator is None
