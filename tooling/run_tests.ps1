param(
    [string]$venvDir = '.venv'
)

$py = Join-Path $venvDir 'Scripts\python.exe'
if (-not (Test-Path $py)) {
    Write-Error "Venv python not found. Run tooling\run_setup.ps1 first."
    exit 1
}

Write-Host "Running pytest with $py"
& $py -m pytest -q
