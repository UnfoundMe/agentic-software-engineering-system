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

## If `dev-up` fails with "password authentication failed for user ases_su"

The container is healthy, the password in `.env` is correct, and it still
fails - this happened during development on a machine with a **native
PostgreSQL service already installed and listening on 5432**. Docker's own
port mapping never gets a chance to bind that port; every connection to
`localhost:5432` from the host silently reaches the *other* Postgres instead,
which has never heard of `ases_su`.

Check with (Windows):

```powershell
Get-NetTCPConnection -LocalPort 5432 -State Listen
Get-Process -Id <OwningProcess>   # look for a process literally named "postgres"
```

If that's the cause, set a different port in `.env` and recreate the
container:

```
POSTGRES_PORT=5433
```

```powershell
docker compose --env-file .env -f infra/docker-compose.yml down -v
.\scripts\dev-up.ps1
```

Redis (6379) does not have this problem in practice - on Windows it is
normally owned by Docker's own relay (`com.docker.backend.exe` /
`wslrelay.exe`), not a native service, so a conflict there would be unusual.
