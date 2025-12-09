import os
import pandas as pd
import pytest
from datetime import datetime, timedelta
from src.validation.walk_forward import evaluate_walk_forward
from src.tuning.optimizers import BayesianTuner


class ToySimulator:
    def __init__(self):
        self.config = {}
        self.stats = {}

    def configure(self, params):
        # params expected to include 'fee' or 'mult'
        self.config.update(params)

    def place_order(self, order, market_price, is_maker=False):
        # deterministic behavior: fee reduces cash directly
        fee = float(self.config.get("fee", 0.0))
        filled = float(order.size)
        avg_price = float(market_price)
        return {"filled_size": filled, "avg_fill_price": avg_price, "fee": fee}


@pytest.mark.asyncio
async def test_tuner_finds_lowest_fee(tmp_path):
    # simple price series and targets causing a single position change
    n = 10
    rng = pd.date_range("2021-01-01", periods=n, freq="min", tz="UTC")
    prices = pd.DataFrame({"timestamp": rng, "close": [100 + i for i in range(n)]})
    # targets: switch to 1 unit from step 1 onwards
    targets = pd.Series([0.0] + [1.0] * (n - 1), index=rng)

    param_space = {"fee": [0.0, 1.0, 2.0]}
    tuner = BayesianTuner(param_space, n_trials=10, seed=0)

    res = evaluate_walk_forward(prices, targets, ToySimulator, window=5, step=5, tuner=tuner, param_space=param_space)
    # single fold expected
    folds = res["folds"]
    assert len(folds) == 1
    best = folds[0]["best_params"]
    assert best is not None
    # best fee should be the smallest (0.0)
    assert float(best.get("fee", 999)) == 0.0
