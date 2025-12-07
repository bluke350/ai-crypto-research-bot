Local tuning example
====================

This project includes a tiny local random-search tuner and a convenience
PowerShell script to run tuning with example data.

How to run (PowerShell):

```powershell
# Run 10 random trials using the sample CSV
.\examples\run_tuning_local.ps1 -PricesCsv examples/sample_prices.csv -NTrials 10 -Seed 0
```

Notes:
- The script sets `PYTHONPATH` so the pipeline imports the local `src` package.
- By default the pipeline will use the `RandomSearchTuner` when `--optimizer random`.
- Results are written to `experiments/artifacts/<run_id>/results.json`.
