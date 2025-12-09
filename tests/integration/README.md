Integration tests: CSV-driven CLI notes
=====================================

This short note explains the CLI options used by the integration tests and how to run them.

- `--prices-csv <path>`: path to a historical prices CSV. The CSV must contain a timestamp column and a `close` column.
- `--timestamp-col <name>`: the name of the timestamp column in the CSV (default: `timestamp`). The CLI will parse this column as datetimes and use it for indexing the prices DataFrame. If omitted, the CLI will try `timestamp`, then fall back to the first column.
- `--pair <pair>`: trading pair string passed to the `ExchangeSimulator` (default: `XBT/USD`). This controls which pair-specific rules (lot size, min notional, fees) are applied during simulation.

Example (PowerShell):

```powershell
$env:WS_DISABLE_HEALTH_SERVER = "1"
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python orchestration/cli.py eval --ckpt models/ml_eval_ckpt.pkl --prices-csv examples/sample_prices.csv --timestamp-col timestamp --pair "XBT/USD" --out experiments/eval
```

Notes for CI and tests
- The integration test sets `PYTHONPATH` so the CLI can import the local `src` package.
- The test also disables the health server via `WS_DISABLE_HEALTH_SERVER=1` to avoid starting background HTTP servers during pytest runs.
- The `--pair` option is used to create an `ExchangeSimulator(pair=<pair>)` instance inside the CLI; tests that expect pair-specific behavior should pass the appropriate `--pair` value.

If you want additional details (example CSV formats, sample checkpoints, or pair rules), I can expand this README.
