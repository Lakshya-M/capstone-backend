# Start FastAPI so phones / ESP32 on your WiFi can reach it (not only localhost).
#
# If you see "running scripts is disabled": double-click  run_api_lan.cmd
# Or run:  powershell -NoProfile -ExecutionPolicy Bypass -File .\run_api_lan.ps1
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "=== Use this IPv4 in esp32_room4_d1.ino  TWIN_POST_URL  (same subnet as ESP32) ===" -ForegroundColor Cyan
Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object {
        $_.IPAddress -notmatch '^127\.' -and
        $_.IPAddress -notmatch '^169\.254\.' -and
        $_.PrefixOrigin -ne 'WellKnown'
    } |
    Sort-Object InterfaceAlias |
    ForEach-Object { Write-Host ("  http://{0}:8000/api/twin/reading   ({1})" -f $_.IPAddress, $_.InterfaceAlias) }
Write-Host ""
Write-Host "Hotspot WiFi is often 'Public' in Windows — firewall may block port 8000." -ForegroundColor Yellow
Write-Host "If ESP32 gets POST -1: run  open_firewall_port8000.ps1  as Administrator (once)." -ForegroundColor Yellow
Write-Host "Then test from this PC:  .\test_lan_api.ps1  (while this window keeps running)." -ForegroundColor DarkGray
Write-Host ""
Write-Host "Starting: uvicorn backend.main:app --host 0.0.0.0 --port 8000" -ForegroundColor Green
Write-Host ""

$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path $py) {
    & $py -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
} else {
    python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
}
