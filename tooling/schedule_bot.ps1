<#
Schedule the bot_service to run on Windows Task Scheduler.

Usage (PowerShell, admin may be required):
  .\tooling\schedule_bot.ps1 -Action Register -TaskName "ai-crypto-bot" -IntervalMinutes 60

This creates a scheduled task that runs the project's venv python to invoke
`python -m scripts.bot_service --once` on the specified interval.
#>

[CmdletBinding()]
param(
    [ValidateSet('Register', 'Unregister')]
    [string]$Action = 'Register',
    [string]$TaskName = 'ai-crypto-bot',
    [int]$IntervalMinutes = 60,
    [string]$WorkingDirectory = "$PSScriptRoot\..",
    [string]$PythonPath = "$PSScriptRoot\..\\.venv\\Scripts\\python.exe"
)

function Register-BotTask {
    param($TaskName, $IntervalMinutes, $WorkingDirectory, $PythonPath)
    $action = New-ScheduledTaskAction -Execute $PythonPath -Argument "-m scripts.bot_service --once"
    $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) -RepetitionDuration ([TimeSpan]::MaxValue)
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances Parallel
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest -Force | Out-Null
    Write-Host "Registered scheduled task '$TaskName' to run every $IntervalMinutes minutes."
}

function Unregister-BotTask {
    param($TaskName)
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Unregistered scheduled task '$TaskName'."
    }
    else {
        Write-Host "Task '$TaskName' not found."
    }
}

if ($Action -eq 'Register') {
    Register-BotTask -TaskName $TaskName -IntervalMinutes $IntervalMinutes -WorkingDirectory $WorkingDirectory -PythonPath $PythonPath
}
else {
    Unregister-BotTask -TaskName $TaskName
}
