"""grant ases_app read access to alembic_version

Revision ID: 2fb5dbe21d8d
Revises: 8b0a48dce088
Create Date: 2026-09-13 02:08:16.873265

Closes a real gap in the previous migration: it granted `ases_app` access to
`control.events` and the four derived views, but never to `alembic_version`
itself - meaning nothing running as `ases_app` could ever check what schema
version is actually applied. That check is exactly the "startup refuses to
run if the Alembic head does not match the database" requirement from
docs/02 Phase 0 (`kernel/store/postgres.py`'s `_ensure_schema_verified`).

A new migration, not an edit to the applied one: `8b0a48dce088` is already
applied to this database, tracked by its revision id, not its file contents.
Editing that file's SQL after the fact would not cause it to be re-run -
Alembic would still believe the database is "at" that revision - so the only
correct way to add a grant that migration should have included is a new
revision on top of it.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "2fb5dbe21d8d"
down_revision: str | Sequence[str] | None = "8b0a48dce088"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("GRANT SELECT ON control.alembic_version TO ases_app")


def downgrade() -> None:
    op.execute("REVOKE SELECT ON control.alembic_version FROM ases_app")
