<#
.SYNOPSIS
    Stop ASES local infrastructure.

.PARAMETER Purge
    Also delete the Postgres volume. THIS DESTROYS THE AUDIT LOG and every
    recorded run. Requires explicit confirmation.
#>
[CmdletBinding()]
param(
    [switch]$Purge
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$compose = @('--env-file', (Join-Path $root '.env'), '-f', (Join-Path $root 'infra\docker-compose.yml'))

if ($Purge) {
    Write-Host 'This deletes the Postgres volume: the event log, all artifacts,' -ForegroundColor Red
    Write-Host 'all approvals and all recorded runs. It cannot be undone.'      -ForegroundColor Red
    $answer = Read-Host "Type 'destroy' to confirm"
    if ($answer -ne 'destroy') { Write-Host 'Aborted.'; exit 1 }
    docker compose @compose down -v
    Write-Host '==> Stopped and volume removed.' -ForegroundColor Yellow
} else {
    docker compose @compose down
    Write-Host '==> Stopped. Data preserved in volume ases_pgdata.' -ForegroundColor Green
}
