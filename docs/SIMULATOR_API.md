Simulator API (explicit constructor params)

Overview
- The `Simulator` now prefers explicit constructor parameters for execution-related behavior instead of an ad-hoc `rules` dict.
- Legacy `rules` is still supported as a fallback, but new code should use the explicit kwargs or structured cost models (`FeeModel`, `SlippageModel`, `LatencySampler`).

New constructor args (high level)
- `partial_fill_fraction: float` — fraction (0..1) of an order that will be filled.
- `partial_fill_slices: int` — number of slices to aggregate multi-part fills into.
- `book_depth: float` — factor representing market depth (deeper book -> lower impact).
- `slippage_k: float` — slippage impact parameter (passed to impact model when not using `SlippageModel`).
- `slippage_daily_vol: float` — daily vol used by impact model when not using `SlippageModel`.
- `maker_bps: int`, `taker_bps: int` — basis points for maker/taker fee fallback.
- `fixed_fee_pct: Optional[float]` — if provided, fee is computed as `abs(notional)*fixed_fee_pct`.
- `fixed_slippage_pct: Optional[float]` — if provided, slippage is `price * fixed_slippage_pct`.
- `fee_model`, `slippage_model`, `latency_model` — preferred structured model objects for deterministic or stochastic costs.

Example usage

```python
from src.execution.simulator import Simulator
from src.execution.cost_models import FeeModel, SlippageModel, LatencySampler

fee = FeeModel(fixed_fee_pct=0.00075)
slip = SlippageModel(fixed_slippage_pct=0.001)
lat = LatencySampler(base_ms=10, jitter_ms=2, seed=0)

sim = Simulator(
    partial_fill_fraction=0.75,
    partial_fill_slices=4,
    book_depth=2.0,
    slippage_k=0.05,
    slippage_daily_vol=0.30,
    maker_bps=10,
    taker_bps=20,
    fee_model=fee,
    slippage_model=slip,
    latency_model=lat,
    seed=42,
)

# place an order
from src.execution.order_models import Order
order = Order(order_id="o1", pair="XBT/USD", side="buy", size=1.0, price=50000.0)
fill = sim.place_order(order, market_price=50000.0)
print(fill)
```

Testing & compatibility
- The change was tested locally: full pytest run passed `246 passed, 0 failed`.
- The PR is incremental: `rules` is still supported as a fallback for external code that hasn't migrated yet.

Recommended next steps
- Migrate any remaining consumers of `rules` to the explicit args or structured models.
- Once external consumers are updated, remove the `rules` fallback in a follow-up, breaking-change PR.

PR: https://github.com/bluke350/ai-crypto-research-bot/pull/3

If you'd like, I can also add a short example in `tools/run_backtest.py` (already present) or update the main `README.md` with a brief example.
