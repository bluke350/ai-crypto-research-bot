<#
Run Optuna tuning workers locally.

Examples:
# Run a single worker in foreground
.
    .\.venv\Scripts\python.exe -m orchestration.pipelines.tuning_pipeline --prices-csv examples\sample_prices_for_cli.csv --optimizer optuna --optuna-db sqlite:///experiments/optuna.db --n-trials 100

# Use this helper to start N background worker processes that point to the same Optuna DB
.
    .\tooling\run_optuna_worker.ps1 -PricesCsv examples\sample_prices_for_cli.csv -OptunaDb sqlite:///C:/workspace/ai-crypto-research-bot/experiments/optuna.db -NTrials 100 -Instances 3 -Background

Parameters:
  -PricesCsv   Path to prices CSV (optional)
  -OptunaDb    Optuna DB URL (sqlite:///path or other DB URL)
  -NTrials     Number of trials per worker (default 100)
  -Instances   Number of worker processes to spawn (default 1)
  -Background  If present, spawn workers as detached processes and log output to files
  -Out         Artifacts root (default experiments/artifacts)
  -LogDir      Directory for logs (default experiments/logs)
#>

param(
    [string]$PricesCsv = "",
    [string]$OptunaDb = "sqlite:///experiments/optuna.db",
    [int]$NTrials = 100,
    [int]$Instances = 1,
    [switch]$Background,
    [string]$Out = "experiments/artifacts",
    [string]$LogDir = "experiments/logs"
)

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$py = Join-Path -Path $PSScriptRoot -ChildPath "..\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    # fallback to system python
    $py = "python"
}

for ($i = 1; $i -le $Instances; $i++) {
    $args = "-m orchestration.pipelines.tuning_pipeline --optimizer optuna --optuna-db `"$OptunaDb`" --n-trials $NTrials --output $Out"
    if ($PricesCsv) { $args += " --prices-csv `"$PricesCsv`"" }

    if ($Background) {
        $stdout = Join-Path $LogDir "optuna_worker_${i}.stdout.log"
        $stderr = Join-Path $LogDir "optuna_worker_${i}.stderr.log"
        Start-Process -FilePath $py -ArgumentList $args -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -NoNewWindow | Out-Null
        Write-Host "Started Optuna worker #$i (background). Logs: $stdout, $stderr"
    }
    else {
        Write-Host "Starting Optuna worker #$i (foreground): $py $args"
        & $py $args
    }
}

Write-Host "Launched $Instances worker(s)."
