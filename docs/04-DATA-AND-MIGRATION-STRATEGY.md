# Plan Addendum — Data Model, Schema Lifecycle & Migration Safety

**Status:** Proposed
**Date:** 2026-09-12
**Supersedes:** `02-ORCHESTRATOR-IMPLEMENTATION-PLAN.md` §6 item 7 (two separate databases)
**Answers:** the isolation question, the migration-caution requirement, and the missing schema-lifecycle definition.

---

## 1. Physical layout — one database, two schemas, two roles

**Revised.** The earlier "two separate databases" requirement was over-engineered. Isolation here is a **privilege** problem, not a **container** problem, and Postgres role grants solve it inside a single database.

```
database: ases
  schema control        owner: ases_control     — orchestrator, durable, audit-grade
  schema workload_test  owner: workload_app     — agent-generated, ephemeral, disposable
```

### 1.1 Why schemas are sufficient

The threat is an agent-generated migration reaching orchestrator data. A separate database blocks that by making the tables unreachable on the connection. A separate schema blocks it just as completely **provided the role cannot reach the schema**:

```sql
REVOKE ALL ON SCHEMA control FROM workload_app;
REVOKE ALL ON ALL TABLES IN SCHEMA control FROM workload_app;
ALTER ROLE workload_app SET search_path = workload_test;
ALTER ROLE workload_app NOSUPERUSER NOCREATEDB NOCREATEROLE;
```

Without `USAGE` on `control`, a fully-qualified `DROP TABLE control.events` fails on permission, not on name resolution. `search_path` alone would not be sufficient — the grant is what does the work. This is the same control the two-database design relied on; the database boundary was never the enforcing mechanism.

Schemas are also **operationally better** here:

- Reset between runs is `DROP SCHEMA workload_test CASCADE; CREATE SCHEMA workload_test AUTHORIZATION workload_app;` — which works from an open connection. `DROP DATABASE` cannot be run while connected to the target, so the two-database design needed a second connection just to reset.
- One connection string base, one container, one backup target. `pg_dump -n control` still isolates the audit log for archival.

### 1.2 What a shared instance genuinely does not protect

Honest limitation: schemas give **integrity** isolation, not **availability** isolation. A runaway agent-generated migration can still exhaust connections or hold locks that affect the control plane. Cheap mitigations, applied to the workload role only:

```sql
ALTER ROLE workload_app SET statement_timeout = '30s';
ALTER ROLE workload_app SET lock_timeout = '5s';
ALTER ROLE workload_app SET idle_in_transaction_session_timeout = '60s';
ALTER ROLE workload_app CONNECTION LIMIT 10;
```

That closes the realistic gap. If this ever ran multi-tenant or in production, separate instances would be the right answer — but for a single-operator prototype, two schemas plus these limits is the correct trade.

---

### 1.3 Bootstrap — who creates what, and when

**Nothing is manual.** A fresh clone reaches a working database with one command. Earlier revisions of this document showed the SQL without saying who executes it; this section closes that gap and is an explicit Phase 0 deliverable.

Three concerns, deliberately separated, because they have different lifetimes:

| Layer | Creates | Mechanism | When it runs |
|---|---|---|---|
| **1. Cluster** | the `ases` database, the Postgres superuser | Postgres container's own env vars | Container first start |
| **2. Bootstrap** | roles, schemas, grants, resource limits | `ases db bootstrap` — **idempotent SQL from the CLI** | Every `dev-up`; safe to re-run |
| **3. Control tables** | `runs`, `events`, `artifacts`, `lineage`, `approvals`, `policy_violations` | `ases db upgrade` — **Alembic** | Every `dev-up`; versioned |

Workload schema objects are layer 4 and never bootstrapped — they arrive only through the gated EF Core pipeline in §4.

#### Why not `/docker-entrypoint-initdb.d/`

The obvious approach is to drop `.sql` files into the Postgres image's init directory. **Rejected**, because those scripts run *only when the data directory is empty*. With a named volume that means they execute on the very first `docker compose up` and **never again** — silently. Edit a grant six weeks later and nothing happens; the only ways to apply it are `docker compose down -v` (destroying the audit log) or hand-running SQL, which is exactly what we are trying to avoid.

`ases db bootstrap` instead uses `CREATE ROLE IF NOT EXISTS`-style guards (`DO $$ ... EXCEPTION WHEN duplicate_object`), `CREATE SCHEMA IF NOT EXISTS`, and unconditional `GRANT`/`REVOKE` — all naturally idempotent. One code path that behaves identically on a fresh volume, an existing volume, and in CI, and that can be integration-tested like any other code.

### 1.4 Compose definition

```yaml
# infra/docker-compose.yml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: ases
      POSTGRES_USER: ases_su
      POSTGRES_PASSWORD: ${POSTGRES_SUPERUSER_PASSWORD}
    ports: ["5432:5432"]
    volumes:
      - ases_pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ases_su -d ases"]
      interval: 2s
      timeout: 3s
      retries: 30

  redis:
    image: redis:7-alpine
    command: ["redis-server", "--appendonly", "no"]
    ports: ["6379:6379"]
    # deliberately no volume — see 1.5

volumes:
  ases_pgdata:
```

```powershell
# scripts/dev-up.ps1   (scripts/dev-up.sh is the POSIX twin)
docker compose --env-file .env -f infra/docker-compose.yml up -d --wait
uv run ases db bootstrap      # idempotent: roles, schemas, grants, limits
uv run ases db upgrade        # Alembic: control-plane tables
```

`--wait` blocks on the healthcheck, so bootstrap cannot race a database that is still starting. That race is the single most common cause of flaky first-run setup, and the healthcheck is what removes it.

### 1.5 Volumes and reset semantics

| Store | Volume | Rationale |
|---|---|---|
| Postgres | **named volume `ases_pgdata`** | The event log is the audit record. It must survive `docker compose down`, container replacement and image upgrades. A bind mount is avoided — on Windows it invites permission and line-ending problems, and Docker Desktop's filesystem translation makes it markedly slower. |
| Redis | **none — ephemeral** | It holds a cache and a job queue, both reconstructible by definition. Persisting a cache across runs would let stale entries leak between demo runs, which is worse than losing it. `appendonly no` makes that explicit rather than accidental. |

Three reset levels, each a distinct command, deliberately increasing in destructiveness:

| Command | Effect | Audit log |
|---|---|---|
| `ases db reset-workload` | `DROP SCHEMA workload_test CASCADE` and recreate + re-grant | **preserved** |
| `ases db reset-control` | Alembic downgrade to base, then upgrade | **destroyed** — prompts for confirmation |
| `docker compose down -v` | destroys the volume | **destroyed** — everything |

`reset-workload` is the one that runs between orchestrator runs. It is hard-coded to the `workload_test` schema name and will refuse any argument, so there is no path by which it reaches `control`.

### 1.6 Connection identities

Four identities, least-privilege, never interchangeable:

| Role | Used by | Privileges |
|---|---|---|
| `ases_su` | `ases db bootstrap` only | superuser; creates roles and schemas. Never used at runtime |
| `ases_control` | Alembic (`ases db upgrade`) | owns `control`; DDL within it |
| `ases_app` | the orchestrator at runtime | DML only, **no DDL**; `INSERT`/`SELECT` on `events`, no `UPDATE`/`DELETE` |
| `workload_app` | agent-generated EF Core | owns `workload_test`; **no `USAGE` on `control`** |

The orchestrator running as `ases_app` with no DDL rights is what makes §3.2's append-only claim hold at runtime: even an application-level bug cannot alter the events table, because the connection has no privilege to.

A fourth database, `ases_test`, is created by bootstrap for the orchestrator's own pytest suite, so kernel tests exercise the real Postgres store rather than a substitute.

### 1.7 Acceptance criterion for Phase 0

> On a machine with Docker and `uv` installed, `git clone && cp .env.example .env && ./scripts/dev-up.sh` yields a working database with the correct roles, schemas, grants and control tables, having executed **zero manual SQL** — and `dev-up` run a second time changes nothing and fails nothing.

The second clause is the one worth testing, because it is what proves the bootstrap is genuinely idempotent rather than accidentally first-run-only.

---

## 2. Two systems, two schema lifecycles

This was missing from the plan entirely. The orchestrator and the workload manage schema in fundamentally different ways, and conflating them would be a design error.

| | Control plane (`control`) | Workload (`workload_test`) |
|---|---|---|
| Language | Python | C# |
| Authored by | **Hand-written by me** | **Agent-generated** |
| Tool | Alembic | EF Core Migrations (code-first) |
| Trust | Trusted | **Untrusted — reviewed and gated** |
| Lifetime | Durable across runs | Dropped and recreated per run |
| Applied by | Explicit `ases db upgrade` | Gated pipeline (§4) |
| Auto-migrate on startup | **Never** | **Never — banned by policy** |

---

## 3. Control-plane schema (`control`)

### 3.1 Tables

| Table | Purpose | Notes |
|---|---|---|
| `runs` | one row per orchestrator run | class, workflow name, status, budgets |
| `events` | **the source of truth** | `seq bigserial`, `run_id`, `event_id uuid`, `type`, `payload jsonb`, `prev_hash`, `hash`, `created_at` |
| `artifacts` | content-addressed artifact store | PK is the content hash; immutable |
| `lineage` | provenance edges | `(artifact_id, input_artifact_id)` |
| `approvals` | human decisions | `artifact_hash`, `decision`, `actor`, `reason`, `revoked_at`, `revoked_reason` |
| `policy_violations` | guardrail interceptions | rule id, node, severity, action taken |
| `alembic_version` | migration state | managed by Alembic |

### 3.2 Append-only enforcement — at the database, not just in code

The tamper-evidence claim in `00-IMPLEMENTATION-PLAN.md` §2.2 is only credible if the database enforces it. Two independent controls:

```sql
-- 1. privilege
GRANT INSERT, SELECT ON control.events TO ases_app;
REVOKE UPDATE, DELETE, TRUNCATE ON control.events FROM ases_app;

-- 2. trigger (defence in depth; survives an accidental re-grant)
CREATE FUNCTION control.reject_mutation() RETURNS trigger AS $$
BEGIN RAISE EXCEPTION 'control.events is append-only'; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER events_append_only
  BEFORE UPDATE OR DELETE ON control.events
  FOR EACH ROW EXECUTE FUNCTION control.reject_mutation();
```

Plus a `UNIQUE` constraint on `prev_hash` so a forked chain is rejected by the database rather than discovered later.

### 3.3 Lifecycle

- Migrations authored in `orchestrator/migrations/versions/`, reviewed like any other code.
- Applied by an explicit `ases db upgrade`. **Never on application startup.**
- On startup the orchestrator *checks* the Alembic head against the database and **refuses to start** if they differ. Fail fast, never silently self-modify.

---

## 4. Workload schema — the cautious migration pipeline

This is the direct answer to "be more cautious while applying migrations."

### 4.1 The governing principle

> **The agent generates a migration. It does not apply one.**
> Generation produces a reviewable artifact; application is a separate, classified, approved, reversible action against a disposable schema.

### 4.2 Pipeline

```
1. ENTITY DESIGN      Implementer agent writes entities + DbContext        (sandbox)
2. MIGRATION GEN      dotnet ef migrations add <Name>                      (sandbox, no DB)
3. SQL MATERIALISE    dotnet ef migrations script --idempotent -o m.sql    (sandbox, no DB)
4. CLASSIFY           static analysis of m.sql -> SAFE | RISKY | DESTRUCTIVE
5. POLICY GATE        DESTRUCTIVE or raw SQL -> block, require approval
6. HUMAN GATE         reviewer sees the SQL, not the C#
7. SNAPSHOT           pg_dump --schema-only -n workload_test  -> rollback artifact
8. APPLY              inside a transaction, with lock/statement timeouts
9. VERIFY             has-pending-model-changes must report none
10. DOWN-TEST         apply Down(), assert schema matches snapshot, re-apply
11. INTEGRATION TESTS dotnet test against the migrated schema
```

Steps 1–4 touch no database at all. The first action with any side effect is step 8, and by then a human has seen the exact SQL.

### 4.3 Operation classification

Static analysis of the generated SQL, classifying every statement:

| Class | Operations | Handling |
|---|---|---|
| **SAFE** (additive) | `CREATE TABLE`, `ADD COLUMN` nullable, `CREATE INDEX CONCURRENTLY`, `ADD CONSTRAINT ... NOT VALID`, `CREATE SEQUENCE` | auto-approve |
| **RISKY** (blocking or rewriting) | `ADD COLUMN NOT NULL` with default, non-concurrent `CREATE INDEX`, `ALTER COLUMN TYPE`, `VALIDATE CONSTRAINT` | flagged in the diff, human sees rationale |
| **DESTRUCTIVE** | `DROP TABLE`, `DROP COLUMN`, `DROP SCHEMA`, `TRUNCATE`, `RENAME`, `DELETE` | **denied by default**, explicit approval required |
| **OPAQUE** | any `migrationBuilder.Sql(...)` raw block | **denied by policy** — this is the hallucinated-`DROP` vector, and an agent has no legitimate need for it in this workload |

Blocking raw SQL is the single highest-value rule here: it removes the entire class of "the model emitted something arbitrary and destructive."

### 4.4 Why this is genuinely safe in Postgres

Postgres has **transactional DDL**. A migration applied inside `BEGIN ... COMMIT` either fully applies or fully rolls back — there is no half-migrated schema. Combined with:

```sql
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
-- migration statements
COMMIT;
```

...a migration that would block or hang aborts cleanly instead of wedging the database. (`CREATE INDEX CONCURRENTLY` cannot run in a transaction; if the classifier sees it, the migration is split and that statement is applied separately with its own approval.)

### 4.5 Rollback and compensation

Three layers, matching the recovery model in `02` Phase 5:

1. **Transaction abort** — statement-level failure, free and automatic.
2. **`Down()` migration** — every migration must have a working `Down()`, proven by the apply → down → re-apply cycle in step 10. This is the registered **compensator** for the `ef_database_update` tool.
3. **Schema drop and restore** — `DROP SCHEMA workload_test CASCADE` then replay from the step-7 snapshot. The always-available fallback, because the schema is disposable by design.

### 4.6 Tool registry classification

| Tool | Idempotent | Side effect | Compensator | Approval |
|---|---|---|---|---|
| `dotnet ef migrations add` | no | `sandbox` | delete migration files | no |
| `dotnet ef migrations script` | yes | `none` | — | no |
| `dotnet ef database update` | **no** | **`external`** | `Down()` migration, else schema restore | **yes** |
| `psql -c <DDL>` | no | `external` | — | **denied — not registered** |

Note the last row: there is no registered tool for arbitrary SQL execution. Under rule C-6 (unknown tool = DENY), an agent that tries has no path.

### 4.7 Runtime schema application — explicitly banned

The generated application **must not** call `Database.Migrate()` or `EnsureCreated()` in `Program.cs`. This is enforced by a policy rule and checked by the static validator.

Reasons, which are also the reasons it is banned in real systems: it races across multiple instances, it runs DDL with application credentials, and it applies schema changes that no human ever reviewed. Instead the application **verifies** at startup that the applied migration matches the expected one and fails fast on mismatch — the same pattern as the control plane in §3.3.

---

## 5. EF Core configuration required by this layout

Because the workload lives in a non-default schema, three things must be true of the generated code, and the validator checks each:

```csharp
modelBuilder.HasDefaultSchema("workload_test");

options.UseNpgsql(connectionString, b =>
    b.MigrationsHistoryTable("__EFMigrationsHistory", "workload_test"));
```

and the connection string must use the `workload_app` role, never `ases_control`.

---

## 6. Changes this addendum makes to the existing plans

| Document | Change |
|---|---|
| `02` §4 Phase 0 | "Two Postgres databases" → one database, two schemas, two roles, with grants and resource limits |
| `02` §4 Phase 0 | Add Alembic and the control-plane schema as an explicit deliverable |
| `02` §4 Phase 0 | Add `ases db bootstrap`, `ases db upgrade`, `ases db reset-workload` and the dev-up scripts as deliverables, with the zero-manual-SQL acceptance criterion (§1.7) |
| `02` §4 Phase 3 | Migration tools registered with the classifications in §4.6; raw-SQL tool deliberately absent |
| `02` §6 item 7 | Superseded |
| `03` §9 | Migration risk row now points here |
| New policy rule | `no_runtime_migration` — bans `Database.Migrate()` / `EnsureCreated()` in generated code |
| New policy rule | `no_raw_sql_migration` — bans `migrationBuilder.Sql(...)` |
