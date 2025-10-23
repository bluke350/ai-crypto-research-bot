import pandas as pd
import numpy as np
from src.features import technical, targets


def make_sample(n=100):
    rng = pd.date_range("2021-01-01", periods=n, freq="T", tz="UTC")
    price = np.cumsum(np.random.randn(n)) + 100.0
    df = pd.DataFrame({
        "timestamp": rng,
        "open": price,
        "high": price + np.random.rand(n),
        "low": price - np.random.rand(n),
        "close": price,
        "volume": np.random.rand(n) * 10,
    })
    return df


def test_technical_runs():
    df = make_sample(200)
    out = technical.all_technical(df)
    assert "rsi_14" in out.columns
    assert any(c.startswith("macd_") for c in out.columns)
    assert any(c.startswith("atr_") for c in out.columns)


def test_targets_triple_barrier():
    df = make_sample(50)
    close = df["close"]
    lbl = targets.triple_barrier_labels(close, pt=0.01, sl=0.005, horizon=10)
    assert len(lbl) == len(close)


def test_rsi_edge_cases():
    df = make_sample(5)
    # small window
    res = technical.rsi(df, window=2)
    assert not res.empty
