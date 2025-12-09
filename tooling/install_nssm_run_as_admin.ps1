<#
Simple helper to (re)install the `ai-crypto-controller` NSSM service
to run the project's venv Python directly. Intended to be right-clicked
and run with "Run as administrator" from Explorer or PowerShell.

Usage: Right-click and Run as Administrator, or run from an elevated
PowerShell session:

    .\tooling\install_nssm_run_as_admin.ps1

This script will:
- Ensure `C:\nssm\nssm.exe` exists
- Create `experiments\logs` directory
- Remove any existing `ai-crypto-controller` service
- Install the service to run the venv Python module `scripts.continuous_controller`
- Configure stdout/stderr capture and rotation
- Start the service and print its status and latest logs
#>

$paramBlock = @'
param(
    [string]$PythonPath
)
'@
Invoke-Expression $paramBlock

function Is-Administrator {
    $current = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $current.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Is-Administrator)) {
    Write-Error "This script must be run as Administrator. Right-click and choose 'Run as administrator'."
    exit 1
}

$svc = 'ai-crypto-controller'
$nssm = 'C:\nssm\nssm.exe'
$wd = (Join-Path $PSScriptRoot '..' | Resolve-Path).Path
$py = $null
if ($PythonPath) {
    $py = $PythonPath
}
else {
    $py = Join-Path $PSScriptRoot '..' | Resolve-Path | ForEach-Object { Join-Path $_ '.venv\Scripts\python.exe' }
}
$stdout = Join-Path $wd 'experiments\logs\ai-crypto-controller.out.log'
$stderr = Join-Path $wd 'experiments\logs\ai-crypto-controller.err.log'
$parms = '-m scripts.continuous_controller --daemon --interval 300 --parallel 1 --artifacts-root "' + (Join-Path $wd 'experiments\artifacts') + '" --prices-csv "' + (Join-Path $wd 'examples\sample_prices_for_cli.csv') + '"'

if (-not (Test-Path $nssm)) {
    Write-Error "Could not find nssm at $nssm. Please install NSSM to C:\\nssm\\nssm.exe or update this script."
    exit 2
}

if (-not $py) {
    Write-Error "Could not resolve a Python executable. Provide -PythonPath or ensure .venv exists in the repo root."
    exit 2
}

if (-not (Test-Path $py)) {
    Write-Error "Resolved python path does not exist: $py. Provide a valid -PythonPath."
    exit 2
}

Write-Host "Resolved python: $py"
Write-Host "Installing service '$svc' to run: $($py) $parms"

New-Item -ItemType Directory -Path (Join-Path $wd 'experiments\logs') -Force | Out-Null

try {
    & $nssm remove $svc confirm 2>$null
}
catch {
    # ignore
}

try {
    & $nssm install $svc $py $parms
    & $nssm set $svc AppDirectory $wd
    & $nssm set $svc AppStdout $stdout
    & $nssm set $svc AppStderr $stderr
    & $nssm set $svc AppRotateFiles 1
    & $nssm set $svc AppRestartDelay 5000
    & $nssm set $svc AppStopMethodSkip 0

    & $nssm start $svc
    Start-Sleep -Seconds 3

    sc.exe query $svc

    Write-Host "`n--- tail stdout (last 200 lines) ---"
    if (Test-Path $stdout) { Get-Content $stdout -Tail 200 } else { Write-Host "missing $stdout" }

    Write-Host "`n--- tail stderr (last 200 lines) ---"
    if (Test-Path $stderr) { Get-Content $stderr -Tail 200 } else { Write-Host "missing $stderr" }

}
catch {
    Write-Error "Failed to install/start service: $_"
    exit 3
}

Write-Host "Done. If the service is not RUNNING, please paste the stderr output here and I'll investigate."
