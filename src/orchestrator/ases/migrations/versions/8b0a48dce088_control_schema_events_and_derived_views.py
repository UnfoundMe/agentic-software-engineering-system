"""control schema: events, and derived views

Revision ID: 8b0a48dce088
Revises:
Create Date: 2026-09-13 01:08:37.443594

Design note - a deliberate deviation from docs/04's literal table list:

docs/04 section 1 names six control tables: runs, events, artifacts, lineage,
approvals, policy_violations. Only `events` is created here as a real,
physical table. `runs`, `artifacts`, `approvals` and `policy_violations` are
created as plain SQL VIEWS over `events` instead.

The reason is the same one the whole design rests on: the event log is the
*only* source of truth. A second, separately-maintained physical table for
"the current approvals" or "the artifacts so far" is a cache that can drift
from the log - exactly the failure mode `RunState = fold(events)` exists to
rule out in the Python kernel. A view cannot drift from its source; it is
recomputed on every query, by construction. Duplicating `kernel/state.py`'s
fold logic in SQL to maintain physical tables would mean two implementations
of "what does this event mean" that could silently disagree.

`lineage` is dropped from this list entirely - not created as a table or a
view. `context/lineage.py` (docs/02 Phase 1) already answers exactly the two
questions a lineage table would: forward and backward walks over
`RunState.artifacts`. That is Python-side, working, and tested; a
SQL-recursive-CTE re-implementation of the same graph walk would be a third
implementation of logic that already exists in two consistent places
(the event log, and the fold over it) rather than a third.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "8b0a48dce088"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE control.events (
            id         BIGSERIAL PRIMARY KEY,
            run_id     UUID NOT NULL,
            seq        BIGINT NOT NULL,
            event_id   UUID NOT NULL,
            type       TEXT NOT NULL,
            actor      JSONB NOT NULL,
            payload    JSONB NOT NULL,
            node_id    TEXT,
            attempt    INT,
            created_at TIMESTAMPTZ NOT NULL,
            prev_hash  TEXT NOT NULL,
            hash       TEXT NOT NULL,
            CONSTRAINT events_run_seq_uq UNIQUE (run_id, seq),
            -- Rejects a forked chain at the database, not merely on the next
            -- verify_chain() call. GENESIS_HASH repeats across every run's
            -- first event, so this must be scoped to (run_id, prev_hash),
            -- never a bare UNIQUE(prev_hash).
            CONSTRAINT events_run_prevhash_uq UNIQUE (run_id, prev_hash),
            CONSTRAINT events_run_eventid_uq UNIQUE (run_id, event_id)
        )
    """)
    op.execute("CREATE INDEX events_run_id_created_at_idx ON control.events (run_id, created_at)")
    op.execute("CREATE INDEX events_type_idx ON control.events (type)")

    # Append-only, enforced at the database - the same guarantee
    # kernel/store/jsonl.py relies on the filesystem for, here relies on
    # Postgres for. Privilege revocation (below) is the first line of
    # defence; this trigger is the second, and survives an accidental
    # re-grant that revocation alone would not.
    op.execute("""
        CREATE FUNCTION control.reject_mutation() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'control.events is append-only: % is not permitted', TG_OP;
        END;
        $$ LANGUAGE plpgsql
    """)
    op.execute("""
        CREATE TRIGGER events_append_only
            BEFORE UPDATE OR DELETE ON control.events
            FOR EACH ROW EXECUTE FUNCTION control.reject_mutation()
    """)

    # ases_app (the orchestrator's runtime identity) may append and read, and
    # nothing else. This grant is what makes the append-only guarantee real
    # at runtime, not just enforced against a role that never tries to
    # violate it - see kernel/store/postgres.py and config.py's app_dsn.
    op.execute("GRANT INSERT, SELECT ON control.events TO ases_app")
    op.execute("GRANT USAGE, SELECT ON SEQUENCE control.events_id_seq TO ases_app")
    op.execute("REVOKE UPDATE, DELETE, TRUNCATE ON control.events FROM ases_app")

    # --- derived views - see the module docstring for why these are views,
    # not tables, and why `lineage` has no SQL-side object at all.

    op.execute("""
        CREATE VIEW control.runs AS
        SELECT
            run_id,
            (payload->>'workflow') AS workflow,
            MIN(created_at) AS created_at
        FROM control.events
        WHERE type = 'run.created'
        GROUP BY run_id, (payload->>'workflow')
    """)

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

    op.execute("""
        CREATE VIEW control.approvals AS
        SELECT
            run_id,
            node_id,
            (payload->>'artifact_hash') AS artifact_hash,
            type AS event_type,
            actor,
            (payload->>'reason') AS reason,
            created_at AS decided_at
        FROM control.events
        WHERE type IN (
            'approval.requested', 'approval.granted', 'approval.rejected', 'approval.revoked'
        )
    """)

    op.execute("""
        CREATE VIEW control.policy_violations AS
        SELECT run_id, node_id, payload, created_at
        FROM control.events
        WHERE type = 'policy.violation'
    """)

    for view in ("runs", "artifacts", "approvals", "policy_violations"):
        op.execute(f"GRANT SELECT ON control.{view} TO ases_app")


def downgrade() -> None:
    for view in ("policy_violations", "approvals", "artifacts", "runs"):
        op.execute(f"DROP VIEW IF EXISTS control.{view}")
    op.execute("DROP TRIGGER IF EXISTS events_append_only ON control.events")
    op.execute("DROP FUNCTION IF EXISTS control.reject_mutation()")
    op.execute("DROP TABLE IF EXISTS control.events")
