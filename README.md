# AI Crypto Research Bot

Minimal scaffold for reproducible research pipelines, experiments registry, and walk-forward validation.

See configs/ for project settings.

## Quick Windows (PowerShell) setup

1. Create venv and install requirements (no Activate.ps1 required):

```powershell
cd C:\path\to\ai-crypto-research-bot
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

2. Run tests:

```powershell
.\.venv\Scripts\python -m pytest -q
```

Or use the helper scripts in `tooling\`:

```powershell
# one-time setup
tooling\run_setup.ps1
# run tests
tooling\run_tests.ps1
```

## Chat recovery & preventive tips

If you use an assistant extension (for example GitHub Copilot Chat) inside VS Code, conversations are often stored per VS Code window or per workspace. If you open a different folder or workspace the assistant may show a different (or empty) chat history.

Quick recovery steps:
- Reopen the original workspace file (for example `ai-crypto-research-bot.code-workspace`) from File → Open Recent.
- Copilot Chat saves sessions under your VS Code global storage. On Windows the files live at `%APPDATA%\\Code\\User\\globalStorage\\emptyWindowChatSessions` and `%APPDATA%\\Code\\User\\globalStorage\\github.copilot-chat` — you can copy the JSON files into your project to preserve them.
- If the extension supports cloud sync, sign into the same account and check the extension's History UI.

Prevent losing chats:
- Export or copy important conversations to a Markdown file inside the repo (useful for reproducibility). Example: copy the JSON to `copilot-session-YYYY.json` or paste a cleaned transcript into `docs/`.
- Keep a dedicated workspace file (`.code-workspace`) and open that instead of the raw folder to preserve workspace-scoped extension state.
- If your extension supports pinning or saving conversations, use it.

## Logging and runtime tips

The project includes a sample `logging.conf` in the repo root. To enable structured logging for local runs:

PowerShell:

```powershell
setx LOG_CFG logging.conf
.\.venv\Scripts\python -m your_module
```

Or configure logging programmatically in your application entrypoint by loading `logging.conf` via `logging.config.fileConfig`.

Operational notes:
- The WS client writes per-minute parquet files to `data/raw/{PAIR}/{YYYYMMDD}/{YYYYMMDDTHHMM}.parquet`.
- Checkpoints are persisted to `data/raw/_ws_checkpoints.json` to allow resuming after restarts.
- For production, run the consumer under a process manager and ensure the log file or central logging sink is configured.

## Pipelines and tuning

- The tuning pipeline supports three optimizers: `bayes` (default), `random`, and `optuna`.
- When using `optuna`, you can pass `--optuna-db sqlite:///path/to/db` to persist the Optuna study (useful for parallel or long-running studies), and `--optuna-pruner` to select a pruner (`median` or `asha`).
- The `auto_train_pipeline` also supports using `optuna` and will accept the same `--optuna-db` and `--optuna-pruner` flags.

## Ensemble mapping (training pipeline)

`training_pipeline` supports an explicit mapping option for ensembles. Use `--ensemble-names` to list model names (comma-separated) and `--ensemble-map` to map names to checkpoint files.

Example:

```bash
python -m orchestration.pipelines.training_pipeline \
  --model ml \
  --out experiments/artifacts \
  --ensemble-names modelA,modelB \
  --ensemble-map modelB=/path/to/modelB_ckpt.pth
```

Behavior:
- If `--ensemble-map` provides `name=path` pairs, those files are copied into the run's `checkpoints/` directory and referenced in the saved `ensemble_weights.json`.
- For names without explicit mappings the pipeline attempts to match produced artifacts by substring, then by regex (if the name looks like a pattern), then by fuzzy matching. As a final fallback it will map artifacts by index when counts align.


## Scheduler & Background helpers

- `tooling/install_task.ps1`: create, list, enable/disable, or delete Windows Scheduled Tasks that run the controller or tuning pipelines. Use `-UseWrapper` and `-WorkingDirectory` to avoid quoting issues when running from Task Scheduler.
- `tooling/run_controller.ps1`: start/stop/status helper that launches the controller as a detached process, records a PID to `experiments/controller.pid`, and supports log rotation, a custom Python path (`-PythonPath`), and automatic wrapper generation (`-UseWrapper`).

- `tooling/install_nssm.ps1`: helper to register `run_controller` as a Windows service using NSSM (Non-Sucking Service Manager). Requires `nssm.exe` to be available on PATH or in `C:\nssm\`.


Recommended quick examples (PowerShell):

Create a safe periodic task (runs controller in `--once` mode every 5 minutes):

```powershell
.\tooling\install_task.ps1 -Create -TaskName "ai-crypto-controller" -Schedule Minute -Modifier 5 `
  -Action '"C:\workspace\ai-crypto-research-bot\.venv\Scripts\python.exe" -m scripts.continuous_controller --once --prices-csv C:\workspace\ai-crypto-research-bot\examples\sample_prices_for_cli.csv --n-trials 5 --artifacts-root C:\workspace\ai-crypto-research-bot\experiments\artifacts' `
  -WorkingDirectory C:\workspace\ai-crypto-research-bot -UseWrapper -Highest -Force
```

Start the controller as a background process (with wrapper and log rotation):

```powershell
.\tooling\run_controller.ps1 -Action Start -PricesCsv examples\sample_prices_for_cli.csv -Parallel 2 -LogDir experiments\logs `
  -PythonPath .\.venv\Scripts\python.exe -MaxLogSizeMB 20 -MaxRotatedFiles 7 -UseWrapper -WorkingDirectory C:\workspace\ai-crypto-research-bot
```

## CSV loader and deduplication

The project uses a centralized CSV loader `src.utils.io.load_prices_csv(path, dedupe=...)` for all price CSV inputs. It:

- Parses a `timestamp` column to UTC datetimes when present.
- Detects duplicate timestamps and optionally deduplicates rows using modes: `none`, `first`, `last`, or `mean`.

Common CLI flags that accept CSV inputs (for example `--prices-csv` or `--replay-csv`) now support a `--dedupe` option in many pipelines. Default behavior is `--dedupe first` which keeps the first row for duplicate timestamps.

If you have CSVs with noisy duplicated timestamps, pass `--dedupe mean` to aggregate numeric columns by mean across duplicates, or `--dedupe last` to keep the latest row.

Example usage (training pipeline):

```bash
python -m orchestration.pipelines.training_pipeline --regime-prices-csv examples/XBT_USD_prices.csv --dedupe mean
```

## Migration notes: Simulator API

- The `Simulator` constructor now prefers explicit execution parameters instead of an ad-hoc `rules` dict. New args include: `partial_fill_fraction`, `partial_fill_slices`, `book_depth`, `slippage_k`, `slippage_daily_vol`, `maker_bps`, `taker_bps`, `fixed_fee_pct`, and `fixed_slippage_pct`.
- Structured cost models are preferred: `FeeModel`, `SlippageModel`, and `LatencySampler` (see `src/execution/cost_models.py`).
- Backwards compatibility: existing callers that pass a `rules` dict will continue to work (fallback), but please update your call sites to use explicit kwargs or structured models.

Quick migration example:

```python
from src.execution.simulator import Simulator
from src.execution.cost_models import FeeModel, SlippageModel, LatencySampler

fee = FeeModel(fixed_fee_pct=0.00075)
slip = SlippageModel(fixed_slippage_pct=0.001)
lat = LatencySampler(base_ms=10, jitter_ms=2, seed=0)

sim = Simulator(partial_fill_fraction=0.75, partial_fill_slices=4, book_depth=2.0,
                slippage_k=0.05, slippage_daily_vol=0.30,
                maker_bps=10, taker_bps=20,
                fee_model=fee, slippage_model=slip, latency_model=lat, seed=42)
```

See `docs/SIMULATOR_API.md` in this branch for more details and examples.


