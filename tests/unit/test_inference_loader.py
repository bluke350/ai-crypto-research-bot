from __future__ import annotations

import os
import json
import pandas as pd
import numpy as np

from src.training.trainer import train_ml
from src.training.inference import ModelWrapper


def _make_prices(n=120):
    idx = pd.date_range("2020-01-01", periods=n, freq="T")
    rng = np.random.default_rng(0)
    returns = rng.normal(scale=0.001, size=n)
    price = 100.0 + np.cumsum(returns)
    return pd.DataFrame({"close": price}, index=idx)


def test_inference_loader_and_metrics(tmp_path):
    save_path = tmp_path / "ckpt.pkl"
    saved = train_ml(data_root="", save=str(save_path), steps=50, seed=0)
    assert os.path.exists(saved)
    metrics_path = os.path.splitext(str(saved))[0] + ".metrics.json"
    assert os.path.exists(metrics_path)
    with open(metrics_path, "r", encoding="utf-8") as mf:
        metrics = json.load(mf)
    assert "mse" in metrics and "mae" in metrics

    prices = _make_prices(120)
    wrapper = ModelWrapper.from_file(str(saved))
    preds = wrapper.predict_returns(prices)
    # prediction length should be n-2 and be a pandas Series
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(prices) - 2
