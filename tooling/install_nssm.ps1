<#
Install / manage a Windows service using NSSM (Non-Sucking Service Manager) that runs the project's `run_controller` helper.

Prerequisites:
- Download NSSM (https://nssm.cc/download) and place `nssm.exe` on the PATH or in `C:\nssm\nssm.exe`.

Usage examples:
  # Check NSSM availability
  .\tooling\install_nssm.ps1 -Action Check

  # Install service (uses run_controller.ps1 -Action Start)
  .\tooling\install_nssm.ps1 -Action Install -ServiceName ai-crypto-controller -ControllerPath C:\workspace\ai-crypto-research-bot\tooling\run_controller.ps1 -WorkingDirectory C:\workspace\ai-crypto-research-bot

  # Start/Stop/Remove
  .\tooling\install_nssm.ps1 -Action Start -ServiceName ai-crypto-controller
  .\tooling\install_nssm.ps1 -Action Stop -ServiceName ai-crypto-controller
  .\tooling\install_nssm.ps1 -Action Remove -ServiceName ai-crypto-controller

Notes:
- NSSM is a third-party utility; this helper does not bundle NSSM but automates common tasks.
- When installing, logs are configured under the provided `-LogDir` (default `C:\workspace\ai-crypto-research-bot\experiments\logs`).
#>

param(
    [ValidateSet('Check','Install','Remove','Start','Stop','Status')]
    [string]$Action = 'Check',
    [string]$ServiceName = 'ai-crypto-controller',
    [string]$ControllerPath = '',
    [string]$WorkingDirectory = '',
    [string]$PythonPath = '',
    [string]$LogDir = '',
    [int]$MaxLogSizeMB = 20
)

function Find-Nssm() {
    $candidates = @()
    if ($env:Path) {
        $candidates += (Get-Command nssm -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)
    }
    $candidates += 'C:\nssm\nssm.exe'
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return (Resolve-Path $c).ProviderPath }
    }
    return $null
}

$nssm = Find-Nssm
if (-not $nssm) {
    if ($Action -ne 'Check') {
        Write-Host 'nssm.exe not found on PATH nor at C:\nssm\nssm.exe. Download from https://nssm.cc/download and place nssm.exe on PATH or in C:\nssm\' -ForegroundColor Yellow
        exit 2
    } else {
        Write-Host 'nssm not found. Install from https://nssm.cc/download and put nssm.exe on PATH or in C:\nssm\' -ForegroundColor Yellow
        exit 0
    }
}

if (-not $LogDir) { $LogDir = Join-Path -Path (Get-Location) -ChildPath 'experiments\logs' }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

switch ($Action) {
    'Check' {
        Write-Host "nssm found: $nssm"
    }
    'Install' {
        if (-not $ControllerPath) { Write-Host 'ControllerPath is required for Install' ; exit 2 }
        if (-not (Test-Path $ControllerPath)) { Write-Host "Controller script not found: $ControllerPath"; exit 2 }
        if (-not $WorkingDirectory) { $WorkingDirectory = (Split-Path $ControllerPath -Parent) }

        $pwsh = Join-Path -Path $env:SystemRoot -ChildPath 'System32\WindowsPowerShell\v1.0\powershell.exe'
        if (-not (Test-Path $pwsh)) { $pwsh = 'powershell.exe' }

        # Build NSSM app parameters to run the wrapper in start mode
        $appArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$ControllerPath`" -Action Start"
        $appDir = $WorkingDirectory

        Write-Host "Installing service '$ServiceName' using nssm: $nssm"
        & $nssm install $ServiceName $pwsh $appArgs

        # set working directory
        & $nssm set $ServiceName AppDirectory $appDir

        # configure stdout/stderr
        $stdout = Join-Path $LogDir "$ServiceName.out.log"
        $stderr = Join-Path $LogDir "$ServiceName.err.log"
        & $nssm set $ServiceName AppStdout $stdout
        & $nssm set $ServiceName AppStderr $stderr
        & $nssm set $ServiceName AppRotateFiles 1
        & $nssm set $ServiceName AppRotateOnline 1

        Write-Host "Service configured. Starting service..."
        & $nssm start $ServiceName
        Write-Host "Service '$ServiceName' installed and started. stdout: $stdout stderr: $stderr"
    }
    'Start' {
        Write-Host "Starting service $ServiceName"
        & $nssm start $ServiceName
    }
    'Stop' {
        Write-Host "Stopping service $ServiceName"
        & $nssm stop $ServiceName
    }
    'Remove' {
        Write-Host "Stopping and removing service $ServiceName"
        & $nssm stop $ServiceName
        Start-Sleep -Seconds 1
        & $nssm remove $ServiceName confirm
        Write-Host "Service removed"
    }
    'Status' {
        Write-Host ("Service status for {0}:" -f $ServiceName)
        & $nssm status $ServiceName
    }
}
