param(
    [string]$PricesCsv = "examples/sample_prices.csv",
    [int]$Steps = 200,
    [int]$Seed = 0,
    [switch]$AutoDeploy
)

$env:WS_DISABLE_HEALTH_SERVER = "1"
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python -m orchestration.auto_run --prices-csv $PricesCsv --steps $Steps --seed $Seed --out-root experiments/artifacts @($([string]($AutoDeploy ? '--auto-deploy' : '')))
