import pandas as pd
import pytest

from src.validation.backtester import run_backtest
from tests.unit.test_backtester import make_simulator


def make_price_series(n=5, start_price=100.0):
    times = pd.date_range("2020-01-01", periods=n, freq="H")
    prices = pd.DataFrame({"timestamp": times, "close": [start_price + i for i in range(n)]})
    return prices


def test_multiple_fills_per_day_and_partial_fraction():
    # Simulator configured to partially fill only some orders — simulate variability
    from src.execution.simulator import Simulator

    # partial fill 50% for first scenario
    sim1 = Simulator(rules={"partial_fill_fraction": 0.5})
    prices = make_price_series(n=4, start_price=100)
    # targets toggle to force multiple orders
    targets = pd.Series([0.0, 2.0, 0.0, 2.0])
    out1 = run_backtest(prices, targets, sim1)
    execs1 = out1["executions"]

    # Expect two executions (at least) and filled_size equals 50% of requested size
    assert len(execs1) >= 2
    for i, row in execs1.iterrows():
        filled = float(row["filled_size"])
        req = float(row.get("requested_size", 1.0))
        assert abs(filled) > 0 and abs(filled) <= abs(req)
        # since partial_fraction is 0.5, absolute filled_size should be approx half of absolute requested_size
        assert abs(filled) == pytest.approx(0.5 * abs(req))


def test_varying_partial_fill_fractions_and_fee_slippage_consistency():
    from src.execution.simulator import Simulator
    from src.execution.fee_models import compute_fee

    # test multiple fractions
    for frac in (0.25, 0.5, 0.75, 1.0):
        sim = Simulator(rules={"partial_fill_fraction": frac, "slippage_daily_vol": 0.6, "slippage_k": 0.1})
        prices = make_price_series(n=3, start_price=200)
        targets = pd.Series([0.0, 1.0, 1.0])
        out = run_backtest(prices, targets, sim)
        execs = out["executions"]
        assert len(execs) >= 1
        row = execs.iloc[0]


        # simulator stores notional as abs(price * filled_size) where price is the market price
        # reconstruct expected notional from requested_notional / requested_size to get the market price used
        req_size = float(row.get("requested_size", 1.0))
        req_notional = float(row.get("requested_notional", abs(req_size)))
        if abs(req_size) > 0:
            implied_price = req_notional / abs(req_size)
        else:
            implied_price = float(prices.iloc[1]["close"])  # fallback
        expected_notional = abs(row["filled_size"]) * implied_price
        assert row["notional"] == pytest.approx(expected_notional, rel=1e-6)

        # fee recorded should equal compute_fee(notional, side, is_maker) within tolerance
        fee_calc = compute_fee(row["notional"], row.get("side", "buy"), row.get("is_maker", False))
        assert row["fee"] == pytest.approx(fee_calc, rel=1e-6)

        # slippage should produce avg_fill_price different from mid price when k>0
        # Check avg_fill_price lies within implied_price +/- reported slippage (with tiny tolerance)
        implied_price = implied_price
        slippage_amt = float(row.get("slippage", 0.0))
        lower = implied_price - abs(slippage_amt) - 1e-8
        upper = implied_price + abs(slippage_amt) + 1e-8
        assert lower <= float(row["avg_fill_price"]) <= upper
        # slippage field should match the signed difference between avg_fill_price and implied_price within tolerance
        assert float(row.get("slippage", 0.0)) == pytest.approx(float(row["avg_fill_price"]) - implied_price, rel=1e-6)


def test_timestamps_and_latency_reflected():
    # ensure execution timestamps are within input price timestamps and latency present
    sim = make_simulator()
    prices = make_price_series(n=6, start_price=50)
    targets = pd.Series([0.0, 1.0, 2.0, 2.0, 1.0, 0.0])
    out = run_backtest(prices, targets, sim)
    execs = out["executions"]
    assert len(execs) >= 1
    for _, row in execs.iterrows():
        ts = row["timestamp"]
        assert ts in list(prices["timestamp"])
        # latency_ms should be non-negative and present
        assert "latency_ms" in row
        assert row["latency_ms"] >= 0
