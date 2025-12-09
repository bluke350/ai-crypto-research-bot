<#
Create a virtual environment and install project requirements using the venv Python executable.
This script uses explicit venv Python to avoid Activate.ps1 / ExecutionPolicy issues.
#>
param(
    [string]$venvDir = '.venv',
    [string]$PythonExe = ''
)

if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
    Write-Host "Creating venv in $venvDir using explicit python: $PythonExe"
    & $PythonExe -m venv $venvDir
} else {
    Write-Host "Creating venv in $venvDir using 'python' from PATH"
    python -m venv $venvDir
}

$py = Join-Path $venvDir 'Scripts\python.exe'
if (-not (Test-Path $py)) {
    Write-Error "python.exe not found in $venvDir/Scripts after venv creation. Provide a valid PythonExe if needed."
    exit 1
}

Write-Host "Upgrading pip and installing requirements using $py"
& $py -m pip install -U pip
& $py -m pip install -r requirements.txt

Write-Host "Setup complete. Use $py to run commands, e.g. $py -m pytest -q"
Write-Host "Optional: configure logging by copying logging.conf to the workspace root or set LOG_CFG env var to point to a config file. Example: setx LOG_CFG logging.conf"
