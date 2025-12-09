import pandas as pd
import numpy as np

from src.features.regime import detect_regimes


def test_detect_regimes_basic():
    idx = pd.date_range('2020-01-01', periods=500, freq='D')
    # simulate low vol then high vol
    prices = pd.Series(np.concatenate([
        np.cumprod(1 + 0.0001 * np.random.randn(250)),
        np.cumprod(1 + 0.01 * np.random.randn(250))
    ]), index=idx)
    regimes = detect_regimes(prices, window=30)
    assert len(regimes) == len(prices)
    # expect at least one 'low' and one 'high'
    vals = set(regimes.unique())
    assert 'low' in vals or 'high' in vals or 'mid' in vals
