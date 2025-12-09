# Example PowerShell script to register a scheduled Task that runs auto_run daily at 02:00
# Usage: edit paths below and run as Administrator

$Action = New-ScheduledTaskAction -Execute 'Powershell.exe' -Argument "-NoProfile -WindowStyle Hidden -Command \"Set-Location -Path 'C:\workspace\ai-crypto-research-bot'; .\.venv\Scripts\Activate.ps1; python -m orchestration.auto_run --out-root 'C:\workspace\ai-crypto-research-bot\experiments\artifacts' --steps 200 --auto-deploy\""
$Trigger = New-ScheduledTaskTrigger -Daily -At 2am
$Principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -RunLevel Highest
Register-ScheduledTask -TaskName "ai-crypto-auto-run" -Action $Action -Trigger $Trigger -Principal $Principal -Description "Daily automated ML train/eval/deploy run for paper trading"
