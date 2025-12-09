from __future__ import annotations

import pandas as pd
import numpy as np

from src.training.order_mapper import adjust_targets_to_lot_and_min_notional


def test_adjust_rounds_to_lot_and_filters_min_notional():
    idx = pd.date_range("2020-01-01", periods=5, freq="T")
    prices = pd.Series([100.0, 100.0, 50.0, 10.0, 200.0], index=idx)
    targets = pd.Series([0.00005, 0.001, 0.2, 0.3, 0.02], index=idx)

    # lot size 0.0001, min_notional 5.0
    adjusted = adjust_targets_to_lot_and_min_notional(targets, prices, lot_size=0.0001, min_notional=5.0)
    # first value rounds to 0 (below lot), second rounds to 0.001, but notional = 0.001*100=0.1 < 5 -> zero
    assert adjusted.iat[0] == 0.0
    assert adjusted.iat[1] == 0.0
    # third: 0.2 units * 50 = 10 -> above min_notional, should round to nearest lot (0.2)
    assert abs(adjusted.iat[2] - 0.2) < 1e-12
    # fourth: 0.3 * 10 = 3 < 5 -> zero
    assert adjusted.iat[3] == 0.0
    # fifth: 0.02 * 200 = 4 < 5 -> zero
    assert adjusted.iat[4] == 0.0
