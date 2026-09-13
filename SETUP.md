# Setup

How to get ASES running locally — first clone on a new machine, and
resuming after infra was stopped.

## Prerequisites

Install these before anything else:

- **Docker** (Desktop or Engine, with `docker compose`)
- **uv** (Python package/venv manager)
- **.NET 10 SDK** — only needed to build/run the generated URL Shortener
  workload, not the orchestrator itself

## First time on a new machine

`.env` is git-ignored (it holds local secrets/passwords), and `.venv` is not
committed either, so a fresh clone needs three steps before anything works:

```powershell
Copy-Item .env.example .env
uv sync --all-extras
.\scripts\dev-up.ps1
```

```bash
cp .env.example .env
uv sync --all-extras
./scripts/dev-up.sh
```

What each step does:

1. **`Copy-Item .env.example .env`** — creates your local config. Default
   passwords are fine for local dev. `ANTHROPIC_API_KEY` can stay blank
   because `ASES_LLM_MODE=replay` (the default) needs no key.
2. **`uv sync --all-extras`** — installs the Python venv from `uv.lock`. You
   cannot skip to `uv run ...` without this on a fresh clone.
3. **`dev-up`** — starts Postgres + Redis containers, waits on the
   healthcheck, then runs `ases db bootstrap` (idempotent roles/schemas/
   grants) and `ases db upgrade` (Alembic migrations).

Running `dev-up` before copying `.env` fails fast on purpose:
`No .env found. Run: Copy-Item .env.example .env`.

**Gotcha:** if the new machine already runs a native Postgres on port 5432,
Docker's mapping will silently connect you to the *wrong* server (looks like
`password authentication failed`). Change `POSTGRES_PORT` in `.env` (e.g.
`5433`) if so.

## Resuming after the Docker container was stopped

No special recovery needed. Postgres data lives in a **named volume**
(`ases_pgdata`), not a bind mount, so it survives `docker compose stop` or
`docker compose down` (everything short of `-Purge` below). Just re-run:

```powershell
.\scripts\dev-up.ps1
```

```bash
./scripts/dev-up.sh
```

This starts the containers (or recreates them against the existing volume)
and re-runs bootstrap + migrations. Both are explicitly idempotent — a
second run must change nothing and fail nothing — so it's always safe to
run `dev-up` again rather than diagnosing what state things are in.

## Stopping

```powershell
.\scripts\dev-down.ps1           # stop, keep data
.\scripts\dev-down.ps1 -Purge    # stop and DESTROY the audit log
```

```bash
./scripts/dev-down.sh            # stop, keep data
./scripts/dev-down.sh --purge    # stop and DESTROY the audit log
```

Only `-Purge` / `--purge` removes the `ases_pgdata` volume. A plain
`dev-down` preserves it, so the next `dev-up` picks up right where you left
off.

## Everyday commands

```powershell
uv run pytest              # full test suite
uv run pytest tests/unit   # unit only, no Postgres required
uv run ruff check .        # lint
uv run mypy                # typecheck
```

`make` targets exist (`make dev-up`, `make check`, etc.) but only delegate
to the same scripts — `make` isn't installed on the primary dev machine
(Windows), so the scripts in `scripts/` are the source of truth.

## See also

- [README.md](README.md) — architecture, project status, invariants
- [docs/06-VALIDATION-GUIDE.md](docs/06-VALIDATION-GUIDE.md) — how to
  independently verify everything the README claims is done
- [docs/04-DATA-AND-MIGRATION-STRATEGY.md](docs/04-DATA-AND-MIGRATION-STRATEGY.md)
  — why bootstrap/migrations are structured this way
