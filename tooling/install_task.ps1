<#
Install or remove a Windows Scheduled Task to run the controller or an Optuna worker.

Usage examples:

# Install a task that runs every 5 minutes (recommended safe pattern: run once and exit)
.\tooling\install_task.ps1 -TaskName "ai-crypto-controller" -Schedule Minute -Modifier 5 -Action '"C:\workspace\ai-crypto-research-bot\.venv\Scripts\python.exe" -m scripts.continuous_controller --once --prices-csv C:\workspace\ai-crypto-research-bot\examples\sample_prices_for_cli.csv --n-trials 5 --artifacts-root C:\workspace\ai-crypto-research-bot\experiments\artifacts' -Create

# Remove an existing task
.\tooling\install_task.ps1 -TaskName "ai-crypto-controller" -Delete

Parameters:
  -TaskName   Name of the Scheduled Task (default: ai-crypto-controller)
  -Schedule   One of Minute | Hourly | Daily | Weekly | Once
  -Modifier   Schedule modifier (e.g., every N minutes when Schedule=Minute)
  -Action     Command line to run (must be quoted as a single string)
  -StartTime  Optional start time HH:mm (defaults to now + 1 minute)
  -Create     Switch to create/install the task
  -Delete     Switch to delete the task
  -RunAsUser  User to run the task as (default: current user)
  -Force      Overwrite existing task if present

This helper wraps `schtasks.exe` and is intended for local, development-first automation.
It runs the provided action using the specified schedule and logs output to Task Scheduler.
Be careful with quoting; include the full path to the python executable to ensure the venv is used.
#>

param(
    [string]$TaskName = "ai-crypto-controller",
    [ValidateSet('Minute', 'Hourly', 'Daily', 'Weekly', 'Once')]
    [string]$Schedule = 'Minute',
    [int]$Modifier = 5,
    [string]$Action = '',
    [string]$StartTime = '',
    [switch]$Create,
    [switch]$Delete,
    [switch]$List,
    [switch]$Enable,
    [switch]$Disable,
    [switch]$Force,
    [string]$RunAsUser = '',
    [switch]$RunAsSystem,
    [switch]$Highest,
    [string]$WorkingDirectory = '',
    [int]$RepeatIntervalMinutes = 0,
    [int]$RepeatDurationMinutes = 0,
    [switch]$UseWrapper
)

function _exit_with([int]$code, [string]$msg) {
    if ($msg) { Write-Host $msg }
    exit $code
}

if ($Delete) {
    Write-Host "Deleting task '$TaskName'..."
    $cmd = "schtasks /Delete /TN `"$TaskName`" /F"
    Write-Host $cmd
    cmd /c $cmd
    _exit_with 0 "Task deleted"
}

if ($List) {
    Write-Host "Listing task '$TaskName'..."
    $cmd = "schtasks /Query /TN `"$TaskName`" /V /FO LIST"
    cmd /c $cmd
    _exit_with 0 "Listed"
}

if ($Disable) {
    Write-Host "Disabling task '$TaskName'..."
    $cmd = "schtasks /Change /TN `"$TaskName`" /DISABLE"
    cmd /c $cmd
    _exit_with 0 "Disabled"
}

if ($Enable) {
    Write-Host "Enabling task '$TaskName'..."
    $cmd = "schtasks /Change /TN `"$TaskName`" /ENABLE"
    cmd /c $cmd
    _exit_with 0 "Enabled"
}

if ($Create -and -not $Action) {
    _exit_with 2 "Action is required when creating a task. Provide full quoted command line via -Action."
}

if ($Create -and -not $StartTime) {
    # schedule to start 1 minute from now
    $st = (Get-Date).AddMinutes(1)
    $StartTime = $st.ToString('HH:mm')
}

$scheduleFlag = ''
switch ($Schedule) {
    'Minute' { $scheduleFlag = "/SC MINUTE /MO $Modifier" }
    'Hourly' { $scheduleFlag = "/SC HOURLY /MO $Modifier" }
    'Daily' { $scheduleFlag = "/SC DAILY /MO $Modifier" }
    'Weekly' { $scheduleFlag = "/SC WEEKLY /MO $Modifier" }
    'Once' { $scheduleFlag = "/SC ONCE /ST $StartTime" }
}

# repetition flags for schtasks: /RI (repeat interval) and /DU (duration)
$repeatFlag = ''
if ($RepeatIntervalMinutes -gt 0) {
    $repeatFlag = "/RI $RepeatIntervalMinutes"
    if ($RepeatDurationMinutes -gt 0) { $repeatFlag += " /DU ${RepeatDurationMinutes}M" }
}

if ($Create) {
    if ($Force) {
        # delete existing task first
        try { cmd /c "schtasks /Delete /TN `"$TaskName`" /F" } catch { }
    }

    # Optionally write a wrapper .cmd if the action is complex and quoting is tricky
    $tr = $Action
    if ($UseWrapper) {
        $wrapperPath = Join-Path -Path (Get-Location) -ChildPath "${TaskName}.cmd"
        Write-Host "Writing wrapper $wrapperPath"
        $content = "@echo off`ncd /d `"$WorkingDirectory`"`n$Action"
        Set-Content -Path $wrapperPath -Value $content -Encoding ASCII
        $tr = $wrapperPath
    }

    # build run-as flags
    $runAsFlag = ''
    if ($RunAsSystem) {
        $runAsFlag = '/RU "SYSTEM"'
    } elseif ($RunAsUser) {
        $runAsFlag = "/RU `"$RunAsUser`""
    } else {
        # default: create task for current user (no explicit /RU)
        $runAsFlag = ''
    }

    $highestFlag = ''
    if ($Highest) { $highestFlag = '/RL HIGHEST' }

    if ($WorkingDirectory -and -not $UseWrapper) {
        # wrap the action to cd into working dir then run command
        $tr = "cmd.exe /C `"cd /d $WorkingDirectory && $Action`""
    }

    $createCmd = "schtasks /Create /TN `"$TaskName`" $scheduleFlag $repeatFlag /TR `"$tr`" /ST $StartTime $runAsFlag $highestFlag /F"
    Write-Host $createCmd
    cmd /c $createCmd

    Write-Host "Task '$TaskName' installed. Use Task Scheduler GUI to inspect details or run history."
}
