<#
.SYNOPSIS
    Bring up ASES local infrastructure and prepare the database.

.DESCRIPTION
    Three layers, in order (see docs/04 section 1.3):
      1. Cluster    - Postgres container creates the `ases` database
      2. Bootstrap  - idempotent roles, schemas, grants, resource limits
      3. Control    - Alembic creates the control-plane tables

    Safe to run repeatedly. A second run must change nothing and fail nothing;
    that is the Phase 0 acceptance criterion.
#>
[CmdletBinding()]
param(
    [switch]$SkipMigrations
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path (Join-Path $root '.env'))) {
    Write-Error "No .env found. Run: Copy-Item .env.example .env"
}

Write-Host '==> Starting Postgres and Redis' -ForegroundColor Cyan
# --wait blocks on the healthcheck, so bootstrap cannot race a starting database.
docker compose --env-file (Join-Path $root '.env') `
               -f (Join-Path $root 'infra\docker-compose.yml') `
               up -d --wait
if ($LASTEXITCODE -ne 0) { throw "docker compose failed" }

if ($SkipMigrations) {
    Write-Host '==> Skipping bootstrap and migrations (-SkipMigrations)' -ForegroundColor Yellow
    exit 0
}

Write-Host '==> Bootstrapping roles, schemas and grants (idempotent)' -ForegroundColor Cyan
uv run ases db bootstrap
if ($LASTEXITCODE -ne 0) { throw "ases db bootstrap failed" }

Write-Host '==> Applying control-plane migrations' -ForegroundColor Cyan
uv run ases db upgrade
if ($LASTEXITCODE -ne 0) { throw "ases db upgrade failed" }

Write-Host '==> Ready.' -ForegroundColor Green
