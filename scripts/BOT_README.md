**AI Trading Bot — Overview & Quick Start**

- **Purpose**: continuously discover promising USD pairs, tune/train models, promote best models, and run realistic paper trading using the repository's execution adapters.

- **Key components added**:
  - `scripts/pair_selector.py`: lightweight pair selection that ranks pairs by simple Sharpe/volatility/volume heuristics.
  - `scripts/bot_service.py`: orchestrator that discovers pairs (or uses selector output), runs tuning (`scripts.continuous_controller`), optionally trains/promotes and runs `orchestration.paper_live.run_live` for realistic paper trading.
  - `tooling/schedule_bot.ps1`: PowerShell helper to register/unregister a Windows Scheduled Task to run the bot periodically.

- **Security & Safety**:
  - Bot defaults to dry-run: use `--execute` to enable training/promote and `--trade paper` to run paper trading.
  - Live trading is gated behind environment flags and explicit user confirmation in the live executor adapter (`ENABLE_LIVE_EXECUTOR` + `CONFIRM_LIVE`).
  - Rotate/revoke API keys stored in `./.env` immediately if those keys were shared.

- **Quick commands**:
  - Dry-run discover + tune: ``.\.venv\Scripts\python.exe -m scripts.bot_service --once --n-trials 3``
  - Full run (train/promote + paper trade top-1): ``.\.venv\Scripts\python.exe -m scripts.bot_service --once --n-trials 10 --execute --top-k 1 --trade paper``
  - Produce pair selection JSON: ``.\.venv\Scripts\python.exe -m scripts.pair_selector --top-k 12``
  - Register scheduled task (PowerShell): ``powershell -File tooling/schedule_bot.ps1 -Action Register -IntervalMinutes 60``

- **Recommended workflow to go live**:
  1. Run selector frequently (hourly) to discover new candidate pairs.
 2. Run bot in dry-run to tune and inspect metrics & artifacts under `experiments/artifacts`.
 3. When satisfied with repeated profitable backtests + paper runs, rotate API keys & enable live executor flags and `--trade live` with caution.

If you want, I can (a) run a dry-run discovery+tuning now, (b) run a full top-1 train+paper run now, or (c) schedule the bot to run hourly using Task Scheduler. Tell me which.
