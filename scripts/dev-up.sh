#!/usr/bin/env bash
# Bring up ASES local infrastructure and prepare the database.
#
# Three layers, in order (see docs/04 section 1.3):
#   1. Cluster    - Postgres container creates the `ases` database
#   2. Bootstrap  - idempotent roles, schemas, grants, resource limits
#   3. Control    - Alembic creates the control-plane tables
#
# Safe to run repeatedly. A second run must change nothing and fail nothing.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "$ROOT/.env" ]]; then
  echo "No .env found. Run: cp .env.example .env" >&2
  exit 1
fi

echo "==> Starting Postgres and Redis"
# --wait blocks on the healthcheck, so bootstrap cannot race a starting database.
docker compose --env-file "$ROOT/.env" -f "$ROOT/infra/docker-compose.yml" up -d --wait

if [[ "${1:-}" == "--skip-migrations" ]]; then
  echo "==> Skipping bootstrap and migrations"
  exit 0
fi

echo "==> Bootstrapping roles, schemas and grants (idempotent)"
uv run ases db bootstrap

echo "==> Applying control-plane migrations"
uv run ases db upgrade

echo "==> Ready."
