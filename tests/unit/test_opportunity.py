import numpy as np
import pandas as pd

from src.features.opportunity import build_basic_features, OpportunityPredictor, AnomalyDetector, ml_rank_universe


def make_dummy_series(n=30):
    idx = pd.date_range('2021-01-01', periods=n, freq='D')
    prices = pd.Series(np.linspace(1.0, 1.0 + 0.1 * (n - 1), n), index=idx)
    vol = pd.Series(np.linspace(10, 20, n), index=idx)
    return pd.DataFrame({'close': prices, 'volume': vol})


def test_build_basic_features_and_ml_rank():
    a = make_dummy_series(30)
    b = make_dummy_series(30).shift(2).bfill()
    data = {'A': a, 'B': b}
    X, syms = build_basic_features(data, window=10)
    assert X.shape[0] == 2
    assert 'A' in syms and 'B' in syms

    # heuristic predictor
    pred = OpportunityPredictor()
    scores = pred.predict(X)
    assert len(scores) == 2

    # anomaly detector (no fit -> all normal by design)
    an = AnomalyDetector()
    flags = an.predict(X)
    assert flags.shape[0] == 2

    ranked = ml_rank_universe(data)
    assert isinstance(ranked, list)
    assert len(ranked) == 2
