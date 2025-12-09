import pandas as pd
import numpy as np

from src.features.technical import realized_volatility, rolling_volume_stats, all_technical
from src.features.targets import triple_barrier_labels, triple_barrier_vectorized, meta_labels


def test_realized_vol_and_volume_stats():
    timestamps = pd.date_range('2021-01-01', periods=100, freq='T', tz='UTC')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
    vols = np.abs(np.random.randn(100)) * 10
    df = pd.DataFrame({'timestamp': timestamps, 'open': prices, 'high': prices, 'low': prices, 'close': prices, 'volume': vols})

    rv = realized_volatility(df, window=10)
    assert not rv.empty
    assert f'rv_10' in rv.columns

    vs = rolling_volume_stats(df, window=10)
    assert not vs.empty
    assert f'vol_mean_10' in vs.columns
    assert f'vol_z_10' in vs.columns

    allf = all_technical(df)
    # check that added columns exist
    assert any(c.startswith('rv_') for c in allf.columns)
    assert any(c.startswith('vol_mean_') for c in allf.columns)


def test_triple_barrier_vectorized_matches_iterative():
    timestamps = pd.date_range('2021-01-01', periods=60, freq='T', tz='UTC')
    prices = pd.Series(100 + np.cumsum(np.random.randn(60) * 0.5), index=timestamps)

    it = triple_barrier_labels(prices, pt=0.01, sl=0.005, horizon=10)
    vec = triple_barrier_vectorized(prices, pt=0.01, sl=0.005, horizon=10)
    # compare elementwise
    assert it.shape == vec.shape
    assert all(it.fillna(0).astype(int).values == vec.fillna(0).astype(int).values)

    meta = meta_labels(prices, pt=0.01, sl=0.005, horizon=10)
    assert 'label' in meta.columns and 'ttt' in meta.columns
