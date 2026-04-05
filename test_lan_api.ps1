# Quick check: is something listening on port 8000 on your Wi-Fi IPv4?
# Run WHILE .\run_api_lan.ps1 is running (in another terminal).
Set-Location $PSScriptRoot

$wifi = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.InterfaceAlias -match 'Wi-Fi|WLAN|Wireless' -and $_.IPAddress -notmatch '^169\.' } |
    Select-Object -First 1

if (-not $wifi) {
    Write-Host "Could not find a Wi-Fi IPv4. Run ipconfig and set `$ip manually below." -ForegroundColor Red
    $ip = Read-Host "Enter PC Wi-Fi IPv4 (e.g. 172.20.10.4)"
} else {
    $ip = $wifi.IPAddress
    Write-Host "Using Wi-Fi IP: $ip ($($wifi.InterfaceAlias))" -ForegroundColor Cyan
}

Write-Host "`nTest-NetConnection $ip port 8000 ..."
$t = Test-NetConnection -ComputerName $ip -Port 8000 -WarningAction SilentlyContinue
if ($t.TcpTestSucceeded) {
    Write-Host "TCP 8000: OPEN (ESP32 should be able to connect if IP matches)." -ForegroundColor Green
} else {
    Write-Host "TCP 8000: FAILED — start API with .\run_api_lan.ps1 and run open_firewall_port8000.ps1 as Admin." -ForegroundColor Red
}

try {
    $h = Invoke-RestMethod -Uri "http://${ip}:8000/health" -Method Get -TimeoutSec 3
    Write-Host "GET /health OK:" $h
} catch {
    Write-Host "GET /health failed:" $_.Exception.Message -ForegroundColor Red
}
