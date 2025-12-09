# Running the Controller & Workers Locally (no Docker)

This document explains safe, local-first ways to run the automation controller and Optuna workers on Windows (PowerShell), including scheduling via Task Scheduler.

Summary

- The project includes a minimal controller PoC at `scripts/continuous_controller.py`. It's dry-run by default and can be enabled to execute training/promotion with `--execute`.
- Optuna tuning can be run locally with `--optimizer optuna --optuna-db sqlite:///path/to/optuna.db`.
- Monitoring/events: controller emits NDJSON events to `experiments/metrics.log` via `src/metrics/emitter.py`.

Quick setup (PowerShell)

```powershell
cd C:\workspace\ai-crypto-research-bot
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Run controller once (dry-run, safe)

```powershell
.\.venv\Scripts\python -m scripts.continuous_controller --once --prices-csv examples\sample_prices_for_cli.csv --n-trials 2
```

Allow controller to execute training/promotions (use with caution)

```powershell
.\.venv\Scripts\python -m scripts.continuous_controller --once --prices-csv examples\sample_prices_for_cli.csv --n-trials 10 --execute
```

Run an Optuna worker locally (foreground)

```powershell
.\.venv\Scripts\python -m orchestration.pipelines.tuning_pipeline --prices-csv examples\sample_prices_for_cli.csv --optimizer optuna --optuna-db sqlite:///experiments/optuna.db --n-trials 100
```

Run multiple Optuna workers (background helper)

- Use `tooling/run_optuna_worker.ps1` to spawn multiple worker processes (background or foreground):

```powershell
.\tooling\run_optuna_worker.ps1 -PricesCsv examples\sample_prices_for_cli.csv -OptunaDb sqlite:///C:/workspace/ai-crypto-research-bot/experiments/optuna.db -NTrials 200 -Instances 3 -Background
```

Scheduling with Task Scheduler (recommended for periodic runs)

- Use `tooling/install_task.ps1` to create a Windows Scheduled Task that runs the controller in `--once` mode every N minutes. Example: create a task that runs every 5 minutes:

```powershell
.\tooling\install_task.ps1 -TaskName "ai-crypto-controller" -Schedule Minute -Modifier 5 -Action '"C:\workspace\ai-crypto-research-bot\.venv\Scripts\python.exe" -m scripts.continuous_controller --once --prices-csv C:\workspace\ai-crypto-research-bot\examples\sample_prices_for_cli.csv --n-trials 5 --artifacts-root C:\workspace\ai-crypto-research-bot\experiments\artifacts' -Create -Force
```

Removing the scheduled task:

```powershell
.\tooling\install_task.ps1 -TaskName "ai-crypto-controller" -Delete
```

Recommended schedules

- Fast periodic checks (collect small amounts of data / short tuning runs): every 1-5 minutes. Use short `--n-trials` and `--once` mode so each scheduled run exits quickly.
- Regular tuning windows (daily): run a longer Optuna study or ensemble evaluation once per day during low-traffic hours (`Schedule Daily` + start time).
- Heavy tuning / training: run manually or via dedicated workers (not Task Scheduler) to avoid overlapping runs.

Example Task Scheduler command (create with wrapper to avoid quoting issues):

```powershell
.	ooling\install_task.ps1 -Create -TaskName "ai-crypto-controller" -Schedule Minute -Modifier 5 \
	-Action '"C:\workspace\ai-crypto-research-bot\.venv\Scripts\python.exe" -m scripts.continuous_controller --once --prices-csv C:\workspace\ai-crypto-research-bot\examples\sample_prices_for_cli.csv --n-trials 5 --artifacts-root C:\workspace\ai-crypto-research-bot\experiments\artifacts' \
	-WorkingDirectory C:\workspace\ai-crypto-research-bot -UseWrapper -Highest -Force
```

Using the `run_controller` helper (background process)

We also provide a lightweight background runner `tooling/run_controller.ps1` which starts the controller as a detached process, records a PID, and rotates logs.

Examples:

Start (background, create wrapper, rotate logs at 20 MB, keep 7 rotated files):

```powershell
.	ooling\run_controller.ps1 -Action Start -PricesCsv examples\sample_prices_for_cli.csv -Parallel 2 -LogDir experiments\logs \
	-PythonPath .\.venv\Scripts\python.exe -MaxLogSizeMB 20 -MaxRotatedFiles 7 -UseWrapper -WorkingDirectory C:\workspace\ai-crypto-research-bot
```

Stop / Status:

```powershell
.	ooling\run_controller.ps1 -Action Stop
.	ooling\run_controller.ps1 -Action Status
```

Notes:

- Prefer `--once` scheduled runs for Task Scheduler to avoid overlapping executions. The controller can also run continuously via `run_controller` when you want a persistent process.
- When using Task Scheduler with `SYSTEM` account (`-RunAsSystem`), artifacts will be written relative to the working directory or absolute paths you pass to the command.
- Monitor `experiments/metrics.log` (NDJSON) and `experiments/artifacts/*/results.json` to validate behavior.

Logs & artifacts

- Artifacts and results: `experiments/artifacts/<run_id>/results.json` and model files.
- Registry DB: `experiments/registry.db` (RunLogger stores runs, metrics, and artifacts).
- Metrics: `experiments/metrics.log` (NDJSON events emitted by the controller).

Safety notes (local-first)

- Controller defaults to dry-run. Do not use `--execute` until you have validated the controller and safety checks.
- Live execution is gated by executor environment variables in the code (e.g., `ENABLE_REAL_LIVE_EXECUTOR` and `CONFIRM_LIVE`). Do not set these to enable real trading unintentionally.
- Archive `experiments/` regularly to avoid disk growth.

````markdown
# Running the Controller & Workers Locally (no Docker)

This document explains safe, local-first ways to run the automation controller and Optuna workers on Windows (PowerShell), including scheduling via Task Scheduler.

Summary

- The project includes a minimal controller PoC at `scripts/continuous_controller.py`. It's dry-run by default and can be enabled to execute training/promotion with `--execute`.
- Optuna tuning can be run locally with `--optimizer optuna --optuna-db sqlite:///path/to/optuna.db`.
- Monitoring/events: controller emits NDJSON events to `experiments/metrics.log` via `src/metrics/emitter.py`.

Quick setup (PowerShell)

```powershell
cd C:\workspace\ai-crypto-research-bot
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Run controller once (dry-run, safe)

```powershell
.\.venv\Scripts\python -m scripts.continuous_controller --once --prices-csv examples\sample_prices_for_cli.csv --n-trials 2
```

Allow controller to execute training/promotions (use with caution)

```powershell
.\.venv\Scripts\python -m scripts.continuous_controller --once --prices-csv examples\sample_prices_for_cli.csv --n-trials 10 --execute
```

Run an Optuna worker locally (foreground)

```powershell
.\.venv\Scripts\python -m orchestration.pipelines.tuning_pipeline --prices-csv examples\sample_prices_for_cli.csv --optimizer optuna --optuna-db sqlite:///experiments/optuna.db --n-trials 100
```

Run multiple Optuna workers (background helper)

- Use `tooling/run_optuna_worker.ps1` to spawn multiple worker processes (background or foreground):

```powershell
.\tooling\run_optuna_worker.ps1 -PricesCsv examples\sample_prices_for_cli.csv -OptunaDb sqlite:///C:/workspace/ai-crypto-research-bot/experiments/optuna.db -NTrials 200 -Instances 3 -Background
```

Scheduling with Task Scheduler (recommended for periodic runs)

- Use `tooling/install_task.ps1` to create a Windows Scheduled Task that runs the controller in `--once` mode every N minutes. Example: create a task that runs every 5 minutes:

```powershell
.\tooling\install_task.ps1 -TaskName "ai-crypto-controller" -Schedule Minute -Modifier 5 -Action '"C:\workspace\ai-crypto-research-bot\.venv\Scripts\python.exe" -m scripts.continuous_controller --once --prices-csv C:\workspace\ai-crypto-research-bot\examples\sample_prices_for_cli.csv --n-trials 5 --artifacts-root C:\workspace\ai-crypto-research-bot\experiments\artifacts' -Create -Force
```

Removing the scheduled task:

```powershell
.\tooling\install_task.ps1 -TaskName "ai-crypto-controller" -Delete
```

Recommended schedules

- Fast periodic checks (collect small amounts of data / short tuning runs): every 1-5 minutes. Use short `--n-trials` and `--once` mode so each scheduled run exits quickly.
- Regular tuning windows (daily): run a longer Optuna study or ensemble evaluation once per day during low-traffic hours (`Schedule Daily` + start time).
- Heavy tuning / training: run manually or via dedicated workers (not Task Scheduler) to avoid overlapping runs.

Example Task Scheduler command (create with wrapper to avoid quoting issues):

```powershell
.	ooling\install_task.ps1 -Create -TaskName "ai-crypto-controller" -Schedule Minute -Modifier 5 `
  -Action '"C:\workspace\ai-crypto-research-bot\.venv\Scripts\python.exe" -m scripts.continuous_controller --once --prices-csv C:\workspace\ai-crypto-research-bot\examples\sample_prices_for_cli.csv --n-trials 5 --artifacts-root C:\workspace\ai-crypto-research-bot\experiments\artifacts' `
  -WorkingDirectory C:\workspace\ai-crypto-research-bot -UseWrapper -Highest -Force
```

Using the `run_controller` helper (background process)

We also provide a lightweight background runner `tooling/run_controller.ps1` which starts the controller as a detached process, records a PID, and rotates logs.

Examples:

Start (background, create wrapper, rotate logs at 20 MB, keep 7 rotated files):

```powershell
.	ooling\run_controller.ps1 -Action Start -PricesCsv examples\sample_prices_for_cli.csv -Parallel 2 -LogDir experiments\logs `
  -PythonPath .\.venv\Scripts\python.exe -MaxLogSizeMB 20 -MaxRotatedFiles 7 -UseWrapper -WorkingDirectory C:\workspace\ai-crypto-research-bot
```

Stop / Status:

```powershell
.	ooling\run_controller.ps1 -Action Stop
.	ooling\run_controller.ps1 -Action Status
```

Notes:

- Prefer `--once` scheduled runs for Task Scheduler to avoid overlapping executions. The controller can also run continuously via `run_controller` when you want a persistent process.
- When using Task Scheduler with `SYSTEM` account (`-RunAsSystem`), artifacts will be written relative to the working directory or absolute paths you pass to the command.
- Monitor `experiments/metrics.log` (NDJSON) and `experiments/artifacts/*/results.json` to validate behavior.

Logs & artifacts

- Artifacts and results: `experiments/artifacts/<run_id>/results.json` and model files.
- Registry DB: `experiments/registry.db` (RunLogger stores runs, metrics, and artifacts).
- Metrics: `experiments/metrics.log` (NDJSON events emitted by the controller).

Safety notes (local-first)

- Controller defaults to dry-run. Do not use `--execute` until you have validated the controller and safety checks.
- Live execution is gated by executor environment variables in the code (e.g., `ENABLE_REAL_LIVE_EXECUTOR` and `CONFIRM_LIVE`). Do not set these to enable real trading unintentionally.
- Archive `experiments/` regularly to avoid disk growth.

Recommended next steps after local validation

1. Integrate real runtime state into `src.operations.safety.check_limits` (exposure, P&L). Ensure controller queries this before promotions.
2. Add a simple monitoring dashboard or alerts (tail `experiments/metrics.log` or push NDJSON to a local aggregator).
3. For heavy parallel tuning, replace SQLite with Postgres for Optuna persistence.


If you want, I can add a `tooling/run_controller.ps1` wrapper (background/foreground options) and an example `task` JSON for Task Scheduler exports.

## NSSM helper (Windows service)

If you prefer running the controller as a Windows service (auto-restart, service management) you can use NSSM (Non-Sucking Service Manager).

1. Download NSSM from https://nssm.cc/download and place `nssm.exe` on your PATH or in `C:\nssm\`.
2. Install the helper service using the included script:

```powershell
.\tooling\install_nssm.ps1 -Action Install -ServiceName ai-crypto-controller -ControllerPath C:\workspace\ai-crypto-research-bot\tooling\run_controller.ps1 -WorkingDirectory C:\workspace\ai-crypto-research-bot
```

Controls:

```powershell
.\tooling\install_nssm.ps1 -Action Start -ServiceName ai-crypto-controller
.\tooling\install_nssm.ps1 -Action Stop  -ServiceName ai-crypto-controller
.\tooling\install_nssm.ps1 -Action Remove -ServiceName ai-crypto-controller
.\tooling\install_nssm.ps1 -Action Check
```

Logs created by NSSM will be under `experiments/logs` by default (stdout/stderr configured by the helper).

Be careful: NSSM is a third-party utility. The helper does not bundle NSSM; follow official download instructions and verify checksums where applicable.

````
