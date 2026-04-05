# Re-launches open_firewall_port8000.ps1 in an elevated PowerShell (UAC prompt).
$target = Join-Path $PSScriptRoot 'open_firewall_port8000.ps1'
Start-Process powershell.exe -Verb RunAs -ArgumentList @(
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', $target
)
