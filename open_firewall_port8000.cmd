@echo off
cd /d "%~dp0"
echo UAC prompt: allow firewall rule for TCP 8000...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0elevate_firewall.ps1"
pause
