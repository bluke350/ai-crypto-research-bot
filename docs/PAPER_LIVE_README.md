**Paper Live Runner**: Short Usage

- **Purpose**: Run the production ingestion + decision stack in real time, but route all orders to a local paper executor (no live orders). This runner uses the same live WS feed, model inference, and execution logic as the live bot; the only difference is the execution layer.

- Example (run with Kraken WS in paper mode):

```powershell
# run with live WS ticks, paper-mode (default)
python -m orchestration.paper_live --ckpt path/to/model.pkl --use-ws --max-ticks 100
```

- To disable paper mode (DANGER: requires env opt-ins):

```powershell
# disable paper-mode (allow live executor if env flags/creds set)
python -m orchestration.paper_live --ckpt path/to/model.pkl --use-ws --no-paper-mode --live-executor
```

**Safe Checklist Before Enabling Real Live Executor**

1. Confirm you understand the risk: enabling live execution can place real orders and move funds.
2. Test locally in paper-mode and inspect `experiments/artifacts/<run_id>/run.log` and `result.json` to verify expected behavior.
3. Use testnet or sandbox API keys where possible.
4. Set explicit environment flags (both required):
   - `ENABLE_REAL_LIVE_EXECUTOR=1`
   - `CONFIRM_LIVE=1`
5. Provide provider and credentials in environment variables (example for ccxt/kraken):
   - `LIVE_EXECUTOR_PROVIDER=ccxt` (or `krakenex`)
   - `EXCHANGE_API_KEY=...`
   - `EXCHANGE_API_SECRET=...`
6. Optionally set `WS_DISABLE_HEALTH_SERVER=1` in CI/tests to avoid starting health HTTP server.
7. Start with low risk limits via CLI flags:
   - `--max-order-notional 100` (max notional per order)
   - `--max-position 1.0` (max absolute position)
   - `--max-loss 1000.0` (global drawdown circuit breaker)
8. Verify `run.log` and `experiments/artifacts/<run_id>/result.json` after a short live run. Only enable for sustained runs after manual verification.

**What to check in `run.log`**

- Confirm the model predictions and requested targets match expectations.
- Confirm that the executor used is `PaperOrderExecutor` unless you explicitly enabled the live executor.
- Check for circuit-breaker activations or rejected orders in the log during the run.
