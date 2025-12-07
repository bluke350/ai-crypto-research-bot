from __future__ import annotations

import pandas as pd
import numpy as np

from src.features.selection import volatility_rank, liquidity_filter, correlation_rank, rank_universe


def make_prices(close_vals, volumes=None, freq='1m'):
    idx = pd.date_range('2021-01-01', periods=len(close_vals), freq=freq, tz='UTC')
    df = pd.DataFrame({'timestamp': idx, 'close': close_vals})
    if volumes is not None:
        df['volume'] = volumes
    return df


def test_volatility_rank_low_high():
    low = make_prices([100.0 + 0.001*i for i in range(200)])
    high = make_prices([100.0 + (0.01 if i%2==0 else -0.01)*(i+1)/100 for i in range(200)])
    vlow = volatility_rank(low, window=50)
    vhigh = volatility_rank(high, window=50)
    assert vhigh > vlow


def test_liquidity_filter():
    df = make_prices([100,101,102], volumes=[1,2,3])
    assert liquidity_filter(df, min_volume=2.0)
    assert not liquidity_filter(df, min_volume=10.0)


def test_correlation_rank_two_and_three_symbols():
    # create two series perfectly correlated vs one independent
    a = np.linspace(100, 110, 100)
    b = a * 1.0
    c = np.linspace(100, 120, 100) + np.random.RandomState(0).randn(100) * 0.1
    df = pd.DataFrame({'A': a, 'B': b, 'C': c}, index=pd.date_range('2021-01-01', periods=100, tz='UTC'))
    corr = correlation_rank(df)
    # A and B should have very high pairwise correlation, C should be less correlated
    rets = df.pct_change().dropna()
    ab_corr = float(rets['A'].corr(rets['B']))
    ac_corr = float(rets['A'].corr(rets['C']))
    assert ab_corr > 0.99
    assert ac_corr < ab_corr
    assert 'C' in corr
    # correlation_rank should equal mean absolute correlation computed directly
    pd_corr = rets.corr().abs()
    expected_A = float(pd_corr['A'].drop('A').mean())
    assert abs(corr['A'] - expected_A) < 1e-8


def test_rank_universe_simple():
    # A: low vol, high vol and high volume; B: high vol high vol; C: low vol low volume
    a = make_prices([100 + 0.001*i for i in range(200)], volumes=[1000]*200)
    b = make_prices([100 + ((-1)**i) * 0.05 * i/100 for i in range(200)], volumes=[1000]*200)
    c = make_prices([100 + 0.001*i for i in range(200)], volumes=[1]*200)
    ranked = rank_universe({'A':a, 'B':b, 'C':c}, min_volume=10.0, vol_window=50)
    # A should be preferred to C (C requires liquidity >10 false)
    assert ranked.index('A') < ranked.index('C')
    assert 'B' in ranked
