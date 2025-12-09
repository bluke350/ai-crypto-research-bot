<#
Update an existing NSSM service to run the repo venv Python directly.
Run this script as Administrator (right-click -> Run as administrator).

Usage:
  Right-click and Run as Administrator, or from an elevated PowerShell:
    .\tooling\nssm_set_python_admin.ps1 -PythonPath 'C:\path\to\.venv\Scripts\python.exe'
#>

param(
    [string]$PythonPath
)

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

if ($PythonPath) {
    $py = $PythonPath
}
else {
    $py = Join-Path $wd '.venv\Scripts\python.exe'
}

if (-not (Test-Path $nssm)) { Write-Error "nssm not found at $nssm"; exit 2 }
if (-not (Test-Path $py)) { Write-Error "Python not found at $py"; exit 2 }

$parms = '-m scripts.continuous_controller --daemon --interval 300 --parallel 1 --artifacts-root "' + (Join-Path $wd 'experiments\artifacts') + '" --prices-csv "' + (Join-Path $wd 'examples\sample_prices_for_cli.csv') + '"'

Write-Host "Current NSSM config (before):"
& $nssm get $svc Application 2>$null
& $nssm get $svc AppParameters 2>$null
& $nssm get $svc AppDirectory 2>$null

Write-Host "Setting NSSM to run: $py $parms"
& $nssm set $svc Application $py
& $nssm set $svc AppParameters $parms
& $nssm set $svc AppDirectory $wd
& $nssm set $svc AppStdout (Join-Path $wd 'experiments\logs\ai-crypto-controller.out.log')
& $nssm set $svc AppStderr (Join-Path $wd 'experiments\logs\ai-crypto-controller.err.log')
& $nssm set $svc AppRotateFiles 1

Write-Host "Restarting service..."
& $nssm restart $svc
Start-Sleep -Seconds 2

sc.exe query $svc

Write-Host "--- tail wrapper stdout ---"
if (Test-Path (Join-Path $wd 'experiments\logs\ai-crypto-controller.out.log')) { Get-Content (Join-Path $wd 'experiments\logs\ai-crypto-controller.out.log') -Tail 200 } else { Write-Host 'missing wrapper stdout' }
Write-Host "--- tail wrapper stderr ---"
if (Test-Path (Join-Path $wd 'experiments\logs\ai-crypto-controller.err.log')) { Get-Content (Join-Path $wd 'experiments\logs\ai-crypto-controller.err.log') -Tail 400 } else { Write-Host 'missing wrapper stderr' }

Write-Host "--- tail controller stdout ---"
if (Test-Path (Join-Path $wd 'experiments\logs\controller.stdout.log')) { Get-Content (Join-Path $wd 'experiments\logs\controller.stdout.log') -Tail 200 } else { Write-Host 'missing controller stdout' }
Write-Host "--- tail controller stderr ---"
if (Test-Path (Join-Path $wd 'experiments\logs\controller.stderr.log')) { Get-Content (Join-Path $wd 'experiments\logs\controller.stderr.log') -Tail 400 } else { Write-Host 'missing controller stderr' }

Write-Host 'Done.'
