from __future__ import annotations

import pandas as pd
import numpy as np

from src.strategies import MeanReversion, Momentum, MovingAverageCrossover


def _build_prices(n=50):
    idx = pd.date_range("2020-01-01", periods=n, freq="T")
    # create a price series that trends up with some noise
    price = 100 + np.linspace(0, 1, n) + np.random.normal(scale=0.1, size=n)
    return pd.DataFrame({"close": price}, index=idx)


def test_mean_reversion_basic():
    prices = _build_prices(60)
    strat = MeanReversion(window=5, threshold=0.5, size=2.0)
    targets = strat.generate_targets(prices)
    assert isinstance(targets, pd.Series)
    assert targets.index.equals(prices.index)
    # values should be in the set {-size, 0, size}
    uniq = set(targets.unique())
    assert all(v in {-2.0, 0.0, 2.0} for v in uniq)


def test_momentum_basic():
    prices = _build_prices(30)
    strat = Momentum(window=3, size=1.5, threshold=0.0)
    targets = strat.generate_targets(prices)
    assert isinstance(targets, pd.Series)
    assert targets.index.equals(prices.index)
    uniq = set(targets.unique())
    assert all(v in {-1.5, 0.0, 1.5} for v in uniq)


def test_moving_average_compatibility():
    prices = _build_prices(40)
    mac = MovingAverageCrossover(short=3, long=7, size=1.0)
    targets = mac.generate_targets(prices)
    assert isinstance(targets, pd.Series)
    assert targets.index.equals(prices.index)
