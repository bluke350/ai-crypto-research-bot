<#
.SYNOPSIS
  Load environment variables from a `.env` file and optionally run a command.

.DESCRIPTION
  This helper reads a simple KEY=VALUE `.env` file and sets those values as
  environment variables in the current PowerShell process. If `-Command` is
  provided, it executes that command (useful to run pipelines with the loaded
  credentials in the same process).

.EXAMPLE
  # Load .env and run the ingest pipeline
  .\tooling\load_env.ps1 -EnvFile .env -Command ".venv\Scripts\python.exe -m orchestration.pipelines.ingest_pipeline --provider kraken --symbols XBT/USD ETH/USD --interval 1m"

#>
param(
    [string]$EnvFile = ".env",
    [string]$Command = ""
)

if (-not (Test-Path $EnvFile)) {
    Write-Warning "Env file '$EnvFile' not found. Create one from .env.example and add keys."
    if ($Command) {
        Write-Host "Running command without loading env: $Command"
        Invoke-Expression $Command
    }
    exit 1
}

Write-Host "Loading env from $EnvFile"
Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq "" -or $line.StartsWith("#")) { return }
    $pair = $line -split "=", 2
    if ($pair.Length -ne 2) { return }
    $k = $pair[0].Trim()
    $v = $pair[1].Trim()
    # remove surrounding quotes
    if ($v.StartsWith('"') -and $v.EndsWith('"')) { $v = $v.Substring(1, $v.Length - 2) }
    if ($v.StartsWith("'") -and $v.EndsWith("'")) { $v = $v.Substring(1, $v.Length - 2) }
    Write-Host "Setting env var: $k"
    Set-Item -Path Env:\$k -Value $v
}

if ($Command) {
    Write-Host "Executing: $Command"
    Invoke-Expression $Command
}
