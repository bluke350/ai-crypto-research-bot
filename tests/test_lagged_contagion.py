import pandas as pd
import numpy as np
from src.features.lagged_contagion import pairwise_lagged_correlation_matrix, build_lagged_features


def test_pairwise_lagged_matrix_simple():
    idx = pd.date_range('2021-01-01', periods=30, freq='D')
    a = pd.Series(np.linspace(1, 2, 30), index=idx)
    b = a.shift(2).bfill()
    d = {'a': a, 'b': b}
    mat = pairwise_lagged_correlation_matrix(d, max_lag=5)
    assert ('a', 'b') in mat
    lag, corr = mat[('a', 'b')]
    assert isinstance(lag, int)
    assert isinstance(corr, float)


def test_build_lagged_features_output():
    idx = pd.date_range('2021-01-01', periods=40, freq='D')
    base = pd.Series(np.linspace(10, 20, 40), index=idx)
    other = base.shift(3).bfill()
    feats = build_lagged_features(base, {'other': other}, max_lag=5)
    assert isinstance(feats, pd.DataFrame)
    assert feats.shape[0] > 0
