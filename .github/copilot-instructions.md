**Overview**

This repository implements reproducible research pipelines for crypto trading (data ingestion, feature building, tuning, training, backtesting, walk-forward). Focus on `orchestration/` pipelines, `configs/` YAML-driven settings, and the `data/` layout used for checkpoints and parquet outputs.

**Quick Commands (Windows / PowerShell)**
- **Setup:** `python -m venv .venv` then `.\.venv\Scripts\python -m pip install -r requirements.txt`
- **Run tests:** `.\.venv\Scripts\python -m pytest -q`
- **Common pipelines:** see VS Code tasks; example: `python -m orchestration.pipelines.ingest_pipeline --provider kraken --symbols XBT/USD ETH/USD --interval 1m`
- **Backtest:** `python -m orchestration.pipelines.backtest_pipeline --exchange kraken`

**Big-picture architecture**
- **Ingestion & Storage:** `orchestration` contains pipelines that ingest market data (WS/REST) and persist minute parquet files under `data/raw/{PAIR}/{YYYYMMDD}/{YYYYMMDDTHHMM}.parquet`. Checkpoint file: `data/raw/_ws_checkpoints.json`.
- **Feature / Training / Tuning:** Pipelines are YAML-configured in `configs/` and orchestrated by modules under `orchestration/pipelines` (examples: `feature_pipeline`, `training_pipeline`, `tuning_pipeline`). `optuna` is supported for tuning (use `--optuna-db` to persist studies).
- **Models & Artifacts:** Trained models and checkpoints live in `models/` and `experiments/artifacts/`. Model filenames follow patterns such as `ppo_*.pth`.
- **Experiment registry:** `experiments/` holds artifacts and outputs from pipelines; treat it as the canonical run directory for reproducibility.

**Project-specific conventions & patterns**
- **YAML-first config:** All pipeline inputs and environment-specific values are defined in `configs/*.yaml` (e.g., `configs/project.yaml`, `configs/data.yaml`). Prefer editing `configs/` rather than hard-coding values.
- **Command flags override configs:** Pipelines accept CLI flags (e.g., `--optuna-db`, `--optuna-pruner`) that override YAML values at runtime.
- **PowerShell helpers:** Reusable helper scripts live in `tooling/` (e.g., `tooling/run_setup.ps1`, `tooling/run_tests.ps1`) — use them for consistent developer setup on Windows.
- **Logging:** `logging.conf` in repo root is the ground-truth config; pipelines and entrypoints load it. Runtime diagnostic strings are captured by SDKs — preserve them when debugging performance or API issues.

**Integration points & external deps to watch**
- **Exchange providers:** The project contains provider-specific code (Kraken) in `orchestration` and `configs/kraken.yaml`. Messages are persisted as parquet to `data/raw` and consumed by downstream pipelines.
- **Optuna:** tuning pipelines can use `optuna` plus an external DB (`sqlite:///path`) for long runs.
- **Local emulator / runtime:** For local development, use `.venv` + PowerShell helpers. The repo assumes a local file-system-based data store (no external DB required by default).

**Patterns to follow when editing code**
- **Preserve YAML-driven inputs:** Add new CLI flags only when they complement (not replace) `configs/*.yaml` entries.
- **Data path stability:** When writing or reading raw data, use the `data/raw/{PAIR}/{YYYYMMDD}` pattern so downstream pipelines find files predictably.
- **Small, focused changes:** Pipelines are composition-oriented — prefer adding small, testable functions to `orchestration/` rather than large monolithic edits.

**Files & locations to reference when coding or debugging**
- `README.md` — setup and high-level runtime notes (PowerShell examples).
- `configs/` — canonical run-time configuration.
- `orchestration/` — pipeline implementations (ingest, feature, tuning, training, backtest, walk-forward).
- `tooling/` — developer helper scripts for consistent runs.
- `data/raw/` — parquet outputs and `_ws_checkpoints.json`.
- `models/` and `experiments/artifacts/` — trained artifacts and checkpoints.

**When writing tests or modifying pipelines**
- Run the smallest relevant pipeline locally (use `--help` to discover flags). Example: `python -m orchestration.pipelines.feature_pipeline --config configs/features.yaml`.
- Use `pytest -q` and prefer adding integration tests under `tests/integration` when changing end-to-end pipeline behavior.

**Typical troubleshooting checklist**
- Verify `configs/*.yaml` values and override with CLI flags if necessary.
- Check `data/raw/_ws_checkpoints.json` if ingestion resumes incorrectly.
- Inspect `logging.conf` and run with `setx LOG_CFG logging.conf` on Windows to ensure proper logging.

If anything here is unclear or you want more detail about a specific area (for example, examples of `orchestration` module shapes or `configs/project.yaml` keys), tell me which section to expand and I will iterate.

**Exact CLI flags (captured from `--help`)**
Below are the exact options printed by each pipeline's `--help`. Use these when updating docs or wiring new CLI flags.

`orchestration.pipelines.tuning_pipeline` (tuning_pipeline.py)
	- `--symbol` `SYMBOL`
	- `--prices-csv` `PRICES_CSV`
	- `--param-space` `PARAM_SPACE`  (JSON string describing param_space)
	- `--n-trials` `N_TRIALS`
	- `--optimizer` `{bayes,random,optuna}`
	- `--optuna-db` `OPTUNA_DB`  (sqlite:///path)
	- `--optuna-pruner` `{median,asha}`
	- `--window` `WINDOW`
	- `--step` `STEP`
	- `--seed` `SEED`
	- `--output` `OUTPUT`
	- `--per-regime`
	- `--register`

`orchestration.pipelines.training_pipeline` (training_pipeline.py)
	- `--model` `{rl,ml}`
	- `--steps` `STEPS`
	- `--save` `SAVE`
	- `--seeds` `SEEDS`  (Comma separated seeds, e.g. `0,1,2`)
	- `--register`
	- `--replay-pair` `REPLAY_PAIR`
	- `--data-root` `DATA_ROOT`
	- `--seed` `SEED`
	- `--out` `OUT`
	- `--action-scale` `ACTION_SCALE`
	- `--obs-mode` `OBS_MODE`

`orchestration.pipelines.walk_forward_pipeline` (walk_forward_pipeline.py)
	- `--symbol` `SYMBOL`
	- `--prices-csv` `PRICES_CSV`
	- `--ppo-checkpoint` `PPO_CHECKPOINT`  (Path to PPO checkpoint to evaluate)
	- `--window` `WINDOW`
	- `--step` `STEP`
	- `--seed` `SEED`
	- `--output` `OUTPUT`

`orchestration.pipelines.auto_train_pipeline` (auto_train_pipeline.py)
	- `--symbol` `SYMBOL`
	- `--prices-csv` `PRICES_CSV`
	- `--optimizer` `{bayes,random,optuna}`
	- `--optuna-db` `OPTUNA_DB`
	- `--optuna-pruner` `{median,asha}`
	- `--param-space` `PARAM_SPACE`
	- `--n-trials` `N_TRIALS`
	- `--seeds` `SEEDS`
	- `--steps` `STEPS`
	- `--window` `WINDOW`
	- `--step` `STEP`
	- `--output` `OUTPUT`
	- `--per-regime`
	- `--register`

**Concrete Examples & Notes (useful when editing or running pipelines)**
- `tuning_pipeline` flags: `--symbol`, `--prices-csv`, `--param-space` (JSON), `--n-trials`, `--optimizer` (bayes|random|optuna), `--optuna-db` (sqlite:///path), `--optuna-pruner` (median|asha), `--window`, `--step`, `--per-regime`, `--register`.
- `walk_forward_pipeline` flags: `--symbol`, `--prices-csv`, `--ppo-checkpoint`, `--window`, `--step`, `--seed`, `--output`.
- `training_pipeline` flags: `--model` (rl|ml), `--steps`, `--save`, `--seeds` (comma list), `--replay-pair`, `--data-root`, `--seed`, `--out`, `--action-scale`, `--obs-mode`, `--register`.

- CSV input expectations: when passing `--prices-csv` to tuning or walk-forward, the CSV should have at minimum `timestamp` (ISO or epoch), `open`, `high`, `low`, `close`, `volume`. If a `timestamp` column exists, pipelines convert it with `pd.to_datetime(..., utc=True)`. Prefer UTC timestamps.

- Config keys that pipelines commonly read:
	- `configs/kraken.yaml`: `kraken.symbols`, `kraken.bar_interval`, `kraken.ws_url`.
	- `configs/project.yaml`: `project.experiment_db`, `project.artifacts_dir`.
	- `configs/data.yaml`: `data.start`, `data.end`, `data.bar_interval`, `data.provider`.

- Optuna usage: pass `--optimizer optuna` and `--optuna-db sqlite:///path/to/study.db`. The code will import `OptunaTuner` only when `optimizer==optuna` and raises a helpful error if `optuna` isn't installed.

- Registering runs/artifacts: many pipelines accept `--register` which will call `src.persistence.db.RunLogger` to persist metadata and artifacts to the experiment DB (default: `experiments/registry.db`). Ensure DB path in `configs/project.yaml` matches expectations.

**Small actionable editing tips**
- When adding new CLI flags, mirror the pattern in `orchestration/pipelines/*`: add to argparse at top of file, pass through to downstream modules (trainers, tuners) or use to control behavior, and include `--register` support only if you also call `RunLogger`.
- When adding a new pipeline that consumes CSV price data, accept `--prices-csv` and call `pd.read_csv(...); pd.to_datetime(..., utc=True)` on `timestamp` if present — this keeps behavior consistent across pipelines.

If you'd like, I can now:
- Add a short sample CSV to `examples/` demonstrating expected columns and UTC timestamps, or
- Update `.github/copilot-instructions.md` further with a cross-reference table mapping pipeline flags to config keys.

I added a sample CSV at `examples/sample_prices_for_cli.csv` you can use to test `--prices-csv` with the pipelines. Example:

```powershell
python -m orchestration.pipelines.tuning_pipeline --prices-csv examples/sample_prices_for_cli.csv
python -m orchestration.pipelines.walk_forward_pipeline --prices-csv examples/sample_prices_for_cli.csv
```

The CSV contains `timestamp` values in ISO-8601 UTC (e.g. `2021-01-01T00:00:00Z`) and columns: `timestamp,open,high,low,close,volume`.

