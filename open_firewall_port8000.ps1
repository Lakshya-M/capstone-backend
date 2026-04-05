# Allow inbound TCP 8000 so ESP32 / other devices on WiFi can reach FastAPI.
# Phone hotspots are often "Public" in Windows — Private-only rules still block them.
#
# Right-click → Run with PowerShell as Administrator, or:
#   Start-Process powershell -Verb RunAs -ArgumentList '-ExecutionPolicy Bypass -File ""$PWD\open_firewall_port8000.ps1""'
#
#Requires -RunAsAdministrator

$ruleName = "Capstone Smart Building API (TCP 8000)"
$existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Rule already exists: $ruleName" -ForegroundColor Yellow
    exit 0
}

New-NetFirewallRule `
    -DisplayName $ruleName `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalPort 8000 `
    -Profile Domain,Private,Public `
    -Description "FastAPI uvicorn for ESP32 twin + dashboard"

Write-Host "OK: inbound TCP 8000 allowed on Domain, Private, and Public profiles." -ForegroundColor Green
Write-Host "Keep the API running with: .\run_api_lan.ps1" -ForegroundColor Cyan
