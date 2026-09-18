# stop.ps1 - stop the Bonsai llama-server started by start.ps1
$ErrorActionPreference = "Stop"
$pidFile = Join-Path $HOME ".bonsai\llama-server.pid"
if (-not (Test-Path $pidFile)) { Write-Host "No PID file at $pidFile - is the server running?"; exit 1 }
$pidVal = [int](Get-Content $pidFile -Raw).Trim()
$proc = Get-Process -Id $pidVal -ErrorAction SilentlyContinue
if ($proc) {
    Stop-Process -Id $pidVal -Force
    Write-Host "[OK] Stopped llama-server (PID $pidVal)"
} else {
    Write-Host "PID $pidVal not running (already stopped)."
}
Remove-Item $pidFile -ErrorAction SilentlyContinue
