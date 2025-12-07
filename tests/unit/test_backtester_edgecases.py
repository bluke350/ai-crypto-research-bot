import pandas as pd
import pytest

from src.validation.backtester import run_backtest
from src.execution.simulator import Simulator
from src.execution.latency import LatencyModel


def make_simulator():
    lat = LatencyModel(base_ms=1, jitter_ms=1, seed=1)
    return Simulator(latency_model=lat)


def test_empty_prices_raises():
    sim = make_simulator()
    prices = pd.DataFrame({"timestamp": [], "close": []})
    targets = pd.Series([])
    with pytest.raises(ValueError):
        run_backtest(prices, targets, sim)


def test_mismatched_lengths_raises():
    sim = make_simulator()
    times = pd.date_range("2020-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"timestamp": times, "close": [100, 101, 102]})
    targets = pd.Series([0.0, 1.0])
    with pytest.raises(ValueError):
        run_backtest(prices, targets, sim)


def test_tiny_delta_no_execution():
    sim = make_simulator()
    times = pd.date_range("2020-01-01", periods=4, freq="D")
    prices = pd.DataFrame({"timestamp": times, "close": [100, 100, 100, 100]})
    # targets all equal -> no change -> no executions
    targets = pd.Series([0.0, 0.0, 0.0, 0.0])
    out = run_backtest(prices, targets, sim)
    assert len(out["executions"]) == 0


def test_negative_or_zero_price_raises():
    sim = make_simulator()
    times = pd.date_range("2020-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"timestamp": times, "close": [100, 0, 102]})
    targets = pd.Series([0.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        run_backtest(prices, targets, sim)


def test_partial_fill_respected():
    # set simulator to partially fill orders (50%) via rules
    from src.execution.simulator import Simulator

    sim = Simulator(rules={"partial_fill_fraction": 0.5})
    times = pd.date_range("2020-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"timestamp": times, "close": [100, 101, 102]})
    targets = pd.Series([0.0, 1.0, 1.0])
    out = run_backtest(prices, targets, sim)
    # one execution should be recorded, but filled_size should be 0.5
    assert len(out["executions"]) >= 1
    assert out["executions"].iloc[0]["filled_size"] == pytest.approx(0.5)


def test_notional_sizing_mode():
    # use sizing_mode='notional' where targets represent notional exposure
    sim = make_simulator()
    times = pd.date_range("2020-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"timestamp": times, "close": [100, 100, 100]})
    # targets as notional exposure: 0 -> 0, 100 -> exposure 100 -> units = 1 at price 100
    targets = pd.Series([0.0, 100.0, 100.0])
    out = run_backtest(prices, targets, sim, sizing_mode="notional")
    # check that an execution occurred and notional recorded is approx 100 for first fill
    assert len(out["executions"]) >= 1
    assert out["executions"].iloc[0]["notional"] == pytest.approx(100.0)
