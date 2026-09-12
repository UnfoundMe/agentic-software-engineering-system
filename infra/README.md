# Infrastructure

Local development infrastructure. Two services, deliberately.

| Service | Used by | Persistence |
|---|---|---|
| `postgres` | **orchestrator** (`control` schema) and **workload** (`workload_test` schema) | named volume `ases_pgdata` |
| `redis` | **workload only** — cache and job queue. Never the kernel. | none, by design |

## Why no `postgres/init/*.sql`

The `postgres/` directory is a placeholder and stays empty.

Mounting SQL into `/docker-entrypoint-initdb.d/` is the usual approach and it
was rejected: those scripts run **only when the data directory is empty**. With
a named volume they execute once, on first start, and then silently stop
applying. Editing a grant later would do nothing, and the only ways to apply it
would be destroying the volume (and the audit log with it) or running SQL by
hand.

Roles, schemas, grants and resource limits are created by `ases db bootstrap`
instead — idempotent, re-runnable, testable, and identical on a fresh volume, an
existing volume, and in CI. See [`docs/04`](../docs/04-DATA-AND-MIGRATION-STRATEGY.md) §1.3.

## Why Redis has no volume

It holds a cache and a job queue, both reconstructible by definition.
Persisting a cache would let stale entries leak between demo runs, which is
worse than losing it. `--appendonly no --save ""` makes that a decision rather
than an accident.

## Why a named volume rather than a bind mount for Postgres

The event log is the audit record and must survive `docker compose down` and
image upgrades. On Windows, bind mounts additionally bring file-permission and
line-ending problems, and Docker Desktop's filesystem translation makes them
markedly slower.

## Usage

```powershell
Copy-Item .env.example .env
.\scripts\dev-up.ps1        # start, bootstrap, migrate  (idempotent)
.\scripts\dev-down.ps1      # stop, keep data
.\scripts\dev-down.ps1 -Purge   # stop and DESTROY the audit log
```

`dev-up` passes `--wait`, so the healthcheck gates bootstrap and it cannot race
a database that is still starting.
