import pytest
import numpy as np
import pandas as pd

from src.validation.walk_forward import evaluate_walk_forward
from src.tuning.optimizers import BayesianTuner
from src.strategies.moving_average import MovingAverageCrossover
from src.execution.simulator import Simulator


def make_synthetic_prices(n=200, seed=0):
    rs = np.random.RandomState(seed)
    t = pd.date_range("2020-01-01", periods=n, freq="min")
    # random walk in price
    p = 100 + np.cumsum(rs.normal(0, 0.1, size=n))
    df = pd.DataFrame({"timestamp": t, "close": p})
    return df


def test_walk_forward_tuning_smoke():
    prices = make_synthetic_prices(n=120)
    # no precomputed targets; we'll pass a strategy factory and param space
    simulator = Simulator(seed=42)

    param_space = {"short": [3, 5], "long": [10, 20], "size": [1.0]}
    tuner = BayesianTuner(param_space, n_trials=10, seed=1)

    res = evaluate_walk_forward(prices=prices, targets=None, simulator=lambda: Simulator(seed=42), window=60, step=20, tuner=tuner, param_space=param_space, strategy_factory=MovingAverageCrossover)
    assert "folds" in res
    for f in res["folds"]:
        assert "metrics" in f and "best_params" in f
