param(
    [string]$PricesCsv = "examples/sample_prices.csv",
    [int]$NTrials = 10,
    [int]$Seed = 0
)

# Example: run local random-search tuning using the project's pipeline
$env:WS_DISABLE_HEALTH_SERVER = "1"
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python -m orchestration.pipelines.tuning_pipeline --prices-csv $PricesCsv --optimizer random --n-trials $NTrials --seed $Seed
