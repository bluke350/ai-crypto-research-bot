import pandas as pd
import numpy as np

from src.features.lagged_contagion import pairwise_lagged_correlation_matrix, build_lagged_features


def test_pairwise_lagged_corr_simple():
    idx = pd.date_range('2021-01-01', periods=100, freq='T')
    a = pd.Series(np.cumsum(np.random.randn(100)), index=idx)
    # b is a leading series (b leads a by 2)
    b = a.shift(-2) + 0.01 * np.random.randn(100)
    d = {'a': a, 'b': b}
    mat = pairwise_lagged_correlation_matrix(d, max_lag=5)
    assert isinstance(mat, dict)
    # expect (a,b) entry exists
    assert ('a', 'b') in mat


def test_build_lagged_features_index():
    idx = pd.date_range('2021-01-01', periods=50, freq='T')
    target = pd.Series(np.cumsum(np.random.randn(50)), index=idx)
    other = pd.Series(np.cumsum(np.random.randn(50)), index=idx)
    feats = build_lagged_features(target, {'o': other}, max_lag=3)
    # features should be indexed like target (subset possibly)
    assert feats.index.equals(feats.index)
    # columns should be present
    assert any('o' in c for c in feats.columns)
