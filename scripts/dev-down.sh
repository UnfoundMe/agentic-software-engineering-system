#!/usr/bin/env bash
# Stop ASES local infrastructure.
#   --purge  also deletes the Postgres volume. THIS DESTROYS THE AUDIT LOG.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE=(--env-file "$ROOT/.env" -f "$ROOT/infra/docker-compose.yml")

if [[ "${1:-}" == "--purge" ]]; then
  echo "This deletes the Postgres volume: the event log, all artifacts," >&2
  echo "all approvals and all recorded runs. It cannot be undone."       >&2
  read -r -p "Type 'destroy' to confirm: " answer
  [[ "$answer" == "destroy" ]] || { echo "Aborted."; exit 1; }
  docker compose "${COMPOSE[@]}" down -v
  echo "==> Stopped and volume removed."
else
  docker compose "${COMPOSE[@]}" down
  echo "==> Stopped. Data preserved in volume ases_pgdata."
fi
