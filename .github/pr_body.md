Purpose
- Centralize CSV price loading to `src/utils/io.load_prices_csv(path, dedupe=...)` to ensure consistent timestamp parsing and handle duplicate timestamps safely.
- Add a `--dedupe` CLI option and update pipelines/tests to use the loader.
- Add small robustness fixes for ML helpers to avoid NotFittedError in test environments.
- Add unit and integration tests for the loader and a basic CI workflow to run tests.

Key changes
- Added: `src/utils/io.py` (new centralized loader with dedupe modes: none|first|last|mean).
- Updated: pipelines and scripts to use loader and accept `--dedupe` where relevant.
- Updated: `src/features/opportunity.py` to fall back to heuristic scoring when sklearn estimators are not fitted (avoids CI failures).
- Tests: added/updated unit tests and an integration smoke for walk-forward.
- Docs: `README.md` updated with `--dedupe` guidance.
- CI: `.github/workflows/ci.yml` — basic pytest workflow.

How to test locally
1. Run the test suite:

```bash
pytest -q
```

2. Run a smoke walk-forward:

```bash
python -m orchestration.pipelines.walk_forward_pipeline --prices-csv examples/sample_prices_for_cli.csv \
  --ppo-checkpoint models/ppo_smoke.pth --seeds 0,1 --output experiments/artifacts/walk_forward_smoke --dedupe first
```

Notes
- Default dedupe mode is `first`. Use `none` to preserve duplicates or `mean` to aggregate numeric columns across duplicate timestamps.
