import pandas as pd
from src.validation.backtester import run_backtest
from src.execution.simulator import Simulator
from src.execution.latency import LatencyModel


def make_simulator() -> Simulator:
    lat = LatencyModel(base_ms=10, jitter_ms=5, seed=42)
    sim = Simulator(latency_model=lat)
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
