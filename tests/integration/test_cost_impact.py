import pandas as pd
import numpy as np
from src.execution.simulator import Simulator
from src.validation.backtester import run_backtest

try:
    from src.execution.cost_models import FeeModel, SlippageModel, LatencySampler
except Exception:
    FeeModel = None
    SlippageModel = None
    LatencySampler = None


def make_prices(n=50, start=100.0):
    idx = pd.date_range("2021-01-01", periods=n, freq="min")
    close = start + np.linspace(0, 1.0, n)
    df = pd.DataFrame({"timestamp": idx, "close": close})
    return df


def test_costs_reduce_final_portfolio():
    prices = make_prices(40)
    # simple target: go to +1 unit at t=5 and hold
    targets = pd.Series([0.0] * len(prices))
    targets.iloc[5:] = 1.0

    # zero-cost simulator
    if FeeModel is not None and SlippageModel is not None and LatencySampler is not None:
        fee0 = FeeModel(fixed_fee_pct=0.0)
        slip0 = SlippageModel(fixed_slippage_pct=0.0)
        lat0 = LatencySampler(base_ms=0, jitter_ms=0)
    else:
        fee0 = slip0 = lat0 = None
    sim0 = Simulator(fee_model=fee0, slippage_model=slip0, latency_model=lat0, seed=1)

    out0 = run_backtest(prices, targets, simulator=sim0)
    final0 = float(out0['pnl'].iloc[-1])

    # with-cost simulator (fees + slippage)
    if FeeModel is not None and SlippageModel is not None and LatencySampler is not None:
        fee1 = FeeModel(fixed_fee_pct=0.001)
        slip1 = SlippageModel(fixed_slippage_pct=0.002)
        lat1 = LatencySampler(base_ms=10, jitter_ms=5, seed=2)
    else:
        fee1 = slip1 = lat1 = None
    sim1 = Simulator(fee_model=fee1, slippage_model=slip1, latency_model=lat1, seed=2)

    out1 = run_backtest(prices, targets, simulator=sim1)
    final1 = float(out1['pnl'].iloc[-1])

    # final portfolio with costs should be strictly less than zero-cost run
    assert final1 < final0
    # stats should reflect fees/slippage for sim1
    stats1 = getattr(sim1, 'stats', {})
    if stats1:
        assert stats1.get('fees', 0.0) > 0.0 or stats1.get('slippage', 0.0) > 0.0

