from __future__ import annotations

import pandas as pd
import numpy as np

from src.strategies import VolatilityBreakout


def test_volatility_breakout_basic():
    # construct a series with a sudden spike to trigger a breakout
    idx = pd.date_range("2020-01-01", periods=30, freq="T")
    base = np.linspace(100, 100.5, 30)
    base[15] = 105.0  # spike at index 15
    prices = pd.DataFrame({"close": base}, index=idx)
    vb = VolatilityBreakout(window=5, threshold=0.01, size=1.0)
    targ = vb.generate_targets(prices)
    assert isinstance(targ, pd.Series)
    assert targ.index.equals(prices.index)
    # expect at least one non-zero around the spike
    assert any(targ.abs() > 0)
