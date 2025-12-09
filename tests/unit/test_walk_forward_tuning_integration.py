import pytest
import numpy as np
import pandas as pd

from src.validation.walk_forward import evaluate_walk_forward
from src.tuning.optimizers import BayesianTuner
from src.strategies.moving_average import MovingAverageCrossover
from src.execution.simulator import Simulator
try:
    from src.execution.cost_models import FeeModel, SlippageModel, LatencySampler
except Exception:
    FeeModel = None
    SlippageModel = None
    LatencySampler = None


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
    # deterministic cost models for testing
    fee = FeeModel(fixed_fee_pct=0.0) if FeeModel is not None else None
    slip = SlippageModel(fixed_slippage_pct=0.0) if SlippageModel is not None else None
    lat = LatencySampler(seed=42) if LatencySampler is not None else None
    simulator = Simulator(seed=42, fee_model=fee, slippage_model=slip, latency_model=lat)

    param_space = {"short": [3, 5], "long": [10, 20], "size": [1.0]}
    tuner = BayesianTuner(param_space, n_trials=10, seed=1)

    sim_factory = lambda: Simulator(seed=42, fee_model=fee, slippage_model=slip, latency_model=lat)
    res = evaluate_walk_forward(prices=prices, targets=None, simulator=sim_factory, window=60, step=20, tuner=tuner, param_space=param_space, strategy_factory=MovingAverageCrossover)
    assert "folds" in res
    for f in res["folds"]:
        assert "metrics" in f and "best_params" in f
