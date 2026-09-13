"""expose artifact content on the artifacts view

Revision ID: b9dac5f510ef
Revises: 2fb5dbe21d8d
Create Date: 2026-09-13 13:17:43.669533

Phase 4 gap, found and closed while wiring the first real agent: `events`
has always carried the full `ARTIFACT_PRODUCED` payload verbatim in its
JSONB `payload` column (nothing decomposes it - `PostgresEventStore` reads
the column back wholesale), so once `kernel/scheduler.py` started including
a `content` key in that payload, the raw event log already had it. This
`control.artifacts` VIEW, however, explicitly projects only named keys out
of `payload`, and `content` was never one of them - so SQL-side access to
an artifact (a future dashboard, a DB audit query) silently could not see it
even though the underlying log always could. `CREATE OR REPLACE VIEW` only
appends a column here; it does not reorder, retype or remove any existing
one, so nothing that already selects from this view by name is affected.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "b9dac5f510ef"
down_revision: str | Sequence[str] | None = "2fb5dbe21d8d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE OR REPLACE VIEW control.artifacts AS
        SELECT
            (payload->>'artifact_hash') AS artifact_hash,
            run_id,
            node_id,
            (payload->>'kind') AS kind,
            created_at AS produced_at,
            COALESCE(payload->'inputs', '[]'::jsonb) AS inputs,
            payload->'content' AS content
        FROM control.events
        WHERE type = 'artifact.produced'
    """)


def downgrade() -> None:
    # Postgres cannot drop a column via CREATE OR REPLACE VIEW (only append
    # one) - removing `content` requires a real DROP + CREATE, not a REPLACE.
    op.execute("DROP VIEW control.artifacts")
    op.execute("""
        CREATE VIEW control.artifacts AS
        SELECT
            (payload->>'artifact_hash') AS artifact_hash,
            run_id,
            node_id,
            (payload->>'kind') AS kind,
            created_at AS produced_at,
            COALESCE(payload->'inputs', '[]'::jsonb) AS inputs
        FROM control.events
        WHERE type = 'artifact.produced'
    """)
    op.execute("GRANT SELECT ON control.artifacts TO ases_app")
