from __future__ import annotations

import pandas as pd
import numpy as np
import os

from src.training.trainer import train_ml
from src.training.adapter import inference_backtest


class DummySimulator:
    def __init__(self):
        self.stats = {"fills": 0}

    def place_order(self, order, market_price, is_maker=False):
        # simulate an immediate full fill
        size = float(order.size)
        self.stats["fills"] += 1
        return {"filled_size": size, "avg_fill_price": market_price, "fee": 0.0}


def _make_prices(n=120):
    idx = pd.date_range("2020-01-01", periods=n, freq="T")
    rng = np.random.default_rng(1)
    returns = rng.normal(scale=0.001, size=n)
    price = 100.0 + np.cumsum(returns)
    return pd.DataFrame({"close": price}, index=idx)


def test_inference_backtest_flow(tmp_path):
    save_path = tmp_path / "ckpt.pkl"
    # train a tiny model
    train_ml(data_root="", save=str(save_path), steps=10, seed=1)
    prices = _make_prices(60)
    sim = DummySimulator()
    result = inference_backtest(str(save_path), prices, sim, method="threshold", size=1.0, threshold=0.0)
    assert "pnl" in result and "executions" in result and "stats" in result
    # executions may be empty if predictions are 0; ensure pnl exists and has same length
    assert len(result["pnl"]) == len(prices)
    assert isinstance(result["executions"], pd.DataFrame)
