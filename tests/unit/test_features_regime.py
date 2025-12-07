import pandas as pd
import numpy as np

from src.features.regime import identify_regimes, regime_periods


def test_identify_regimes_simple():
    idx = pd.date_range('2021-01-01', periods=200, freq='1min', tz='UTC')
    # low vol then high vol
    low = 100 + np.cumsum(np.random.RandomState(0).randn(100) * 0.001)
    high = low[-1] + np.cumsum(np.random.RandomState(1).randn(100) * 0.02)
    arr = np.concatenate([low, high])
    s = pd.Series(arr, index=idx)
    labels = identify_regimes(s, window=20, high_vol_threshold=0.005)
    rets = s.pct_change().fillna(0.0)
    vol = rets.rolling(window=20, min_periods=1).std()
    mean_first = float(vol.iloc[:100].mean())
    mean_last = float(vol.iloc[100:].mean())
    assert mean_last > mean_first * 5


def test_regime_periods_compresses():
    idx = pd.date_range('2021-01-01', periods=10, freq='1min', tz='UTC')
    labels = pd.Series([0,0,1,1,1,0,0,1,1,0], index=idx)
    periods = regime_periods(labels)
    # expect periods capture start/end correctly and count equals number of regimes (changes +1)
    assert len(periods) == 5
    assert periods[0][2] == 0
    assert periods[1][2] == 1
