import numpy as np
import pandas as pd
from src.features.volume_profile import volume_profile, vpoc, vpin
from src.features.multitimeframe import resample_ohlcv, multitimeframe_fusion
from src.models.ensemble_manager import EnsembleManager


def test_volume_profile_basic():
    idx = pd.date_range('2021-01-01', periods=10, freq='T')
    prices = pd.Series(np.linspace(100, 110, 10), index=idx)
    vols = pd.Series(np.ones(10), index=idx)
    edges, vol_per_bin = volume_profile(prices, vols, bins=5)
    assert len(edges) == 6
    assert vol_per_bin.sum() == 10.0
    v = vpoc(edges, vol_per_bin)
    assert isinstance(v, float)


def test_vpin_returns_series():
    idx = pd.date_range('2021-01-01', periods=50, freq='T')
    vols = pd.Series(np.random.rand(50) * 10.0, index=idx)
    r = vpin(vols, window=10)
    assert isinstance(r, pd.Series)
    assert len(r) == 50


def test_resample_and_fusion():
    idx = pd.date_range('2021-01-01', periods=60, freq='T')
    df = pd.DataFrame({'open': np.linspace(100, 120, 60), 'high': np.linspace(100, 121, 60), 'low': np.linspace(99, 119, 60), 'close': np.linspace(100, 120, 60), 'volume': np.ones(60)}, index=idx)
    res = resample_ohlcv(df, rule='5T')
    assert isinstance(res, pd.DataFrame)
    fused = multitimeframe_fusion(df, [res])
    assert 'hf0_open' in fused.columns


def test_ensemble_manager_basic():
    em = EnsembleManager(['m1', 'm2'])
    preds = {'m1': np.array([0.1, 0.2, 0.3]), 'm2': np.array([0.0, 0.5, -0.1])}
    out = em.predict(preds)
    assert out.shape == (3,)
    # update weights favoring m2
    em.update_weights({'m1': 0.1, 'm2': 1.0}, decay=0.5)
    ws = em.get_weights()
    assert 'm1' in ws and 'm2' in ws
    assert abs(sum(ws.values()) - 1.0) < 1e-6
