import pandas as pd
from src.validation.backtester import run_backtest
from src.execution.simulator import Simulator
from src.execution.latency import LatencyModel


def make_simulator() -> Simulator:
    try:
        from src.execution.cost_models import FeeModel, SlippageModel, LatencySampler
    except Exception:
        FeeModel = None
        SlippageModel = None
        LatencySampler = None
    lat = LatencyModel(base_ms=10, jitter_ms=5, seed=42)
    fee = FeeModel(fixed_fee_pct=0.0) if FeeModel is not None else None
    slip = SlippageModel(fixed_slippage_pct=0.0) if SlippageModel is not None else None
    # prefer LatencySampler if available, otherwise use LatencyModel
    lat_model = LatencySampler(seed=42) if LatencySampler is not None else lat
    sim = Simulator(fee_model=fee, slippage_model=slip, latency_model=lat_model)
    return sim


def test_backtester_smoke() -> None:
    # simple monotonic price series
    times = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"timestamp": times, "close": [100, 102, 101, 103, 105]})
    # targets: start flat, go long 1 unit, hold, reduce to 0, go short -1
    targets = pd.Series([0.0, 1.0, 1.0, 0.0, -1.0])

    sim = make_simulator()
    out = run_backtest(prices, targets, sim, initial_cash=10_000.0)

    assert "pnl" in out and hasattr(out["pnl"], "iloc")
    assert "executions" in out
    # there should be 3 execution events (index 1 buy, index 3 sell to zero, index 4 sell to short)
    assert len(out["executions"]) == 3
    # final portfolio value numeric (sanity check)
    assert not pd.isna(out["pnl"].iloc[-1])
