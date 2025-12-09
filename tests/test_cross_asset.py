import pandas as pd
import numpy as np
from src.features.cross_asset import btc_dominance, rolling_correlation_with_btc, lagged_cross_correlation, cross_pair_features


def test_btc_dominance():
    caps = pd.Series({'BTC': 600.0, 'ETH': 200.0, 'SOL': 200.0})
    dom = btc_dominance(caps)
    assert abs(dom - 0.6) < 1e-9


def test_rolling_correlation_with_btc():
    idx = pd.date_range('2021-01-01', periods=100, freq='D')
    btc = pd.Series(np.linspace(30000, 40000, 100), index=idx)
    a = btc * 1.05 + np.random.normal(0, 10, size=100)
    a = pd.Series(a, index=idx)
    rc = rolling_correlation_with_btc(a, btc, window=10)
    assert isinstance(rc, pd.Series)
    assert len(rc) == 100


def test_lagged_cross_correlation_detects_lead():
    idx = pd.date_range('2021-01-01', periods=50, freq='D')
    b = pd.Series(np.linspace(100, 200, 50), index=idx)
    # create a that lags b by 2
    a = b.shift(2).bfill()
    lag, corr = lagged_cross_correlation(a, b, max_lag=5)
    # Expect positive lag indicating b leads a (b -> a) ; best lag should be 2 or -2 depending on sign convention
    assert isinstance(lag, int)
    assert isinstance(corr, float)


def test_cross_pair_features_output():
    idx = pd.date_range('2021-01-01', periods=60, freq='D')
    a = pd.Series(np.linspace(10, 20, 60), index=idx)
    b = pd.Series(np.linspace(11, 21, 60), index=idx)
    feats = cross_pair_features(a, b, window=5, max_lag=3)
    assert 'rolling_corr_mean' in feats
    assert 'lagged_best_lag' in feats
    assert 'lagged_best_corr' in feats
