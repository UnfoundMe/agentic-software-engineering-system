"""Idempotent database bootstrap: roles, schemas, grants, resource limits.

This is the code behind `ases db bootstrap` (docs/04 section 1.3). It exists
because the obvious alternative - SQL dropped into
`/docker-entrypoint-initdb.d/` - only ever runs once, on an empty volume, and
then silently stops applying. Everything here uses `IF NOT EXISTS` guards or
tolerates the "already exists" error, so running it against a fresh volume, an
already-bootstrapped one, or from CI produces the same result: nothing to do,
or the missing pieces created, never a failure from re-running it.

Connects as the Postgres superuser - the only identity ever used here, and the
only one ever used to run this module. Every other identity created by it
(`ases_control`, `ases_app`, `workload_app`) is deliberately less privileged
than the one running it.
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from ases.config import CONTROL_SCHEMA, WORKLOAD_SCHEMA, Settings

logger = logging.getLogger(__name__)

#: Resource limits applied to the workload role only. Schemas give integrity
#: isolation (the workload role has no USAGE on `control` at all); these give
#: the availability isolation schemas cannot - a runaway agent-generated
#: migration cannot hold a connection, a lock, or an idle transaction
#: indefinitely. Docs/04 section 1.2.
_WORKLOAD_LIMITS: tuple[str, ...] = (
    "ALTER ROLE workload_app SET statement_timeout = '30s'",
    "ALTER ROLE workload_app SET lock_timeout = '5s'",
    "ALTER ROLE workload_app SET idle_in_transaction_session_timeout = '60s'",
    "ALTER ROLE workload_app CONNECTION LIMIT 10",
)


def _create_role_sql(role: str, password: str) -> str:
    # CREATE ROLE has no IF NOT EXISTS in Postgres; DO-block + exception
    # handler is the standard idiom for an idempotent role creation.
    escaped_password = password.replace("'", "''")
    return f"""
    DO $$
    BEGIN
        CREATE ROLE {role} LOGIN PASSWORD '{escaped_password}';
    EXCEPTION WHEN duplicate_object THEN
        ALTER ROLE {role} LOGIN PASSWORD '{escaped_password}';
    END
    $$;
    """


async def bootstrap(settings: Settings) -> None:
    """Create everything docs/04 section 1 requires, idempotently."""
    engine = create_async_engine(settings.superuser_dsn, isolation_level="AUTOCOMMIT")
    try:
        async with engine.begin() as conn:
            logger.info("creating roles")
            await conn.execute(
                text(
                    _create_role_sql(
                        "ases_control", settings.ases_control_password.get_secret_value()
                    )
                )
            )
            await conn.execute(
                text(_create_role_sql("ases_app", settings.ases_app_password.get_secret_value()))
            )
            await conn.execute(
                text(
                    _create_role_sql(
                        "workload_app", settings.workload_app_password.get_secret_value()
                    )
                )
            )

            logger.info("creating schemas")
            await conn.execute(
                text(f"CREATE SCHEMA IF NOT EXISTS {CONTROL_SCHEMA} AUTHORIZATION ases_control")
            )
            await conn.execute(
                text(f"CREATE SCHEMA IF NOT EXISTS {WORKLOAD_SCHEMA} AUTHORIZATION workload_app")
            )

            logger.info("granting connect and schema usage")
            await conn.execute(
                text(f"GRANT CONNECT ON DATABASE {settings.postgres_db} TO ases_control")
            )
            await conn.execute(
                text(f"GRANT CONNECT ON DATABASE {settings.postgres_db} TO ases_app")
            )
            await conn.execute(
                text(f"GRANT CONNECT ON DATABASE {settings.postgres_db} TO workload_app")
            )
            await conn.execute(text(f"GRANT USAGE ON SCHEMA {CONTROL_SCHEMA} TO ases_app"))
            # Table/view-level grants to ases_app (INSERT+SELECT on events,
            # SELECT on the derived views) are issued by the Alembic migration
            # itself, run as ases_control - the owner of those objects. They
            # cannot be granted here because the objects do not exist yet.

            logger.info("isolating the workload role from the control schema")
            await conn.execute(text(f"REVOKE ALL ON SCHEMA {CONTROL_SCHEMA} FROM workload_app"))
            await conn.execute(text("ALTER ROLE workload_app NOSUPERUSER NOCREATEDB NOCREATEROLE"))
            await conn.execute(text(f"ALTER ROLE workload_app SET search_path = {WORKLOAD_SCHEMA}"))

            logger.info("applying workload resource limits")
            for stmt in _WORKLOAD_LIMITS:
                await conn.execute(text(stmt))
    finally:
        await engine.dispose()


async def reset_workload(settings: Settings) -> None:
    """Drop and recreate `workload_test`. The audit log (`control`) is
    untouched - this is the reset that runs between orchestrator runs
    (docs/04 section 1.5), hard-coded to this one schema name so there is no
    path by which it could ever reach `control`.
    """
    engine = create_async_engine(settings.superuser_dsn, isolation_level="AUTOCOMMIT")
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP SCHEMA IF EXISTS {WORKLOAD_SCHEMA} CASCADE"))
            await conn.execute(text(f"CREATE SCHEMA {WORKLOAD_SCHEMA} AUTHORIZATION workload_app"))
    finally:
        await engine.dispose()
