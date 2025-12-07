Paper-mode / Safety Controls

Overview

This project supports a paper trading mode that mirrors live trading behavior but never sends real orders to an exchange. Paper-mode uses a deterministic/stochastic simulator and an `OrderExecutor` abstraction so the rest of the system can be agnostic to whether orders are simulated or live.

Key components

- `src/execution/order_executor.py` — defines `OrderExecutor`, `PaperOrderExecutor`, and `LiveOrderExecutor`.
  - `PaperOrderExecutor` delegates to `ExchangeSimulator` and records fills in `executor.fills`.
  - `LiveOrderExecutor` is a dry-run placeholder; implementors must replace it with a safe adapter to real exchange APIs.

- `src/execution/simulators.py` — exchange rules and partial-fill / slippage simulation.

- `src/validation/backtester.py` — `run_backtest` now accepts either a `simulator` or an `executor` (preferred). Use `executor` to route orders through `PaperOrderExecutor`.

- `orchestration/auto_run.py` — adds `--paper-mode` flag to run in paper mode and persists executor-recorded fills into `result.json` for audit.

Safety recommendations

- Always run with `--paper-mode` when connecting to live market data for testing.
- Implement `LiveOrderExecutor` carefully and default it to `dry_run=True` until approved.
- Add a kill-switch (process-wide flag) and enforce hard limits:
  - Max position size per pair
  - Max per-order notional
  - Max daily loss (circuit breaker)
  - Rate-limits between orders

- Require manual approval / multi-sig for any model promotion to live deployment. Use the AI policy only as a gating suggestion until thoroughly validated.

Observability

- `result.json` contains `pnl` and `executions` for each run; when using `--paper-mode` the executor-recorded fills are appended for replay and audit.
- Add structured logs and metrics (Prometheus) for fills, rejections, and errors.

Replayability

- `orchestration/cli.py` includes a `replay` subcommand that can re-run the original `auto_run` using a saved `repro.json` for deterministic replay (preserves seeds and CLI args).

Credentials

- Prefer storing exchange credentials in a local `.env` file (do not check this into source control).
- A small PowerShell helper is available at `tooling/load_env.ps1` to load `.env` entries into the current session and run a command with those environment variables in place. Example:

  `./tooling/load_env.ps1 -EnvFile .env -Command ".\.venv\Scripts\python.exe -m orchestration.pipelines.ingest_pipeline --provider kraken --symbols XBT/USD ETH/USD --interval 1m"`

Next steps

- Implement a real `LiveOrderExecutor` that integrates with exchange SDKs behind feature flags and secure credentials.
- Add unit and integration tests that run end-to-end against streaming CSV/WS inputs in `--paper-mode` as part of CI.
