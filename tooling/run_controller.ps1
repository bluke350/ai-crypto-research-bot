<#
Run the controller as a background process, manage PID and logs.

Usage:
  Start controller in background (default):
    .\tooling\run_controller.ps1 -Action Start -PricesCsv examples\sample_prices_for_cli.csv -Parallel 2 -LogDir experiments\logs

  Stop controller:
    .\tooling\run_controller.ps1 -Action Stop

  Check status:
    .\tooling\run_controller.ps1 -Action Status

Options:
  -Action  Start|Stop|Status
  -PricesCsv  Path to CSV (optional)
  -Parallel  Number of tuning workers
  -Execute   If present, pass --execute to controller (enable training)
  -LogDir    Directory to write stdout/stderr logs (default experiments/logs)
  -PidFile   Path to PID file (default experiments/controller.pid)

This helper uses Start-Process for a detached run and records the PID. It does not
install services; use `tooling/install_task.ps1` or NSSM helper for production runs.
#>

param(
    [ValidateSet('Start', 'Stop', 'Status')]
    [string]$Action = 'Start',
    [string]$PricesCsv = '',
    [int]$Parallel = 1,
    [switch]$Execute,
    [string]$LogDir = 'experiments/logs',
    [string]$PidFile = 'experiments/controller.pid',
    [string]$PythonPath = '',
    [int]$MaxLogSizeMB = 10,
    [int]$MaxRotatedFiles = 5,
    [switch]$UseWrapper,
    [string]$WorkingDirectory = ''
)

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

if ($PythonPath) {
    $py = $PythonPath
}
else {
    $py = Join-Path -Path $PSScriptRoot -ChildPath "..\.venv\Scripts\python.exe"
    if (-not (Test-Path $py)) { $py = 'python' }
}

function Rotate-Log($path, $maxMB, $maxFiles) {
    if (-not (Test-Path $path)) { return }
    $sizeMB = (Get-Item $path).Length / 1MB
    if ($sizeMB -lt $maxMB) { return }

    $ts = Get-Date -Format "yyyyMMddHHmmss"
    $dir = Split-Path $path -Parent
    $base = Split-Path $path -Leaf
    $rotated = Join-Path $dir "$($base).$ts"
    Rename-Item -Path $path -NewName $rotated

    # prune older rotated files
    $pattern = "$base.*"
    $files = Get-ChildItem -Path $dir -Filter $pattern | Where-Object { $_.Name -ne $base } | Sort-Object LastWriteTime -Descending
    if ($files.Count -gt $maxFiles) {
        $files[$maxFiles..($files.Count - 1)] | ForEach-Object { Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue }
    }
}

function _is_running($pid) {
    try { Get-Process -Id $pid -ErrorAction Stop | Out-Null; return $true } catch { return $false }
}

switch ($Action) {
    'Start' {
        if (Test-Path $PidFile) {
            $existing = Get-Content $PidFile -ErrorAction SilentlyContinue
            if ($existing -and (_is_running [int]$existing)) { Write-Host "Controller already running (PID $existing)"; exit 0 }
        }

        $args = "-m scripts.continuous_controller --parallel $Parallel --artifacts-root experiments/artifacts"
        if ($PricesCsv) { $args += " --prices-csv `"$PricesCsv`"" }
        if ($Execute) { $args += " --execute" }

        $stdout = Join-Path $LogDir "controller.stdout.log"
        $stderr = Join-Path $LogDir "controller.stderr.log"

        # rotate logs if they exceed size
        Rotate-Log $stdout $MaxLogSizeMB $MaxRotatedFiles
        Rotate-Log $stderr $MaxLogSizeMB $MaxRotatedFiles

        $execPath = $py
        $execArgs = $args
        if ($UseWrapper) {
            # create a wrapper .cmd in the working directory or repo root
            if (-not $WorkingDirectory) { $WorkingDirectory = (Get-Location).ProviderPath }
            $wrapper = Join-Path $WorkingDirectory "run_controller_wrapper.cmd"
            $wrapperContent = "@echo off`ncd /d `"$WorkingDirectory`"`n`"$py`" $args %*"
            Set-Content -Path $wrapper -Value $wrapperContent -Encoding ASCII
            $execPath = $wrapper
            $execArgs = ''
        }

        $startParams = @{
            FilePath               = $execPath
            WindowStyle            = 'Hidden'
            RedirectStandardOutput = $stdout
            RedirectStandardError  = $stderr
            PassThru               = $true
        }
        if ($execArgs) { $startParams['ArgumentList'] = $execArgs }
        if ($WorkingDirectory) { $startParams['WorkingDirectory'] = $WorkingDirectory }

        $proc = Start-Process @startParams
        $proc.Id | Out-File -FilePath $PidFile -Encoding ascii
        Write-Host "Started controller PID $($proc.Id). Logs: $stdout, $stderr"
    }
    'Stop' {
        if (-not (Test-Path $PidFile)) { Write-Host "No PID file found at $PidFile"; exit 0 }
        $controllerPid = Get-Content $PidFile | ForEach-Object { $_.Trim() } | Select-Object -First 1
        if (-not $controllerPid) { Write-Host "PID file empty"; Remove-Item $PidFile -Force; exit 0 }
        try {
            Stop-Process -Id [int]$controllerPid -Force -ErrorAction Stop
            Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
            Write-Host "Stopped controller PID $controllerPid"
        }
        catch {
            Write-Host (("Failed to stop process {0}: {1}" -f $controllerPid, $_))
        }
    }
    'Status' {
        if (-not (Test-Path $PidFile)) { Write-Host "No PID file found"; exit 0 }
        $controllerPid = Get-Content $PidFile | ForEach-Object { $_.Trim() } | Select-Object -First 1
        if ($controllerPid -and (_is_running [int]$controllerPid)) { Write-Host "Controller running (PID $controllerPid)" } else { Write-Host "Controller not running" }
    }
}
