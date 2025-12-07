from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - sklearn may be optional in some envs
    RandomForestRegressor = None
    IsolationForest = None
    Pipeline = None
    StandardScaler = None


def build_basic_features(prices_by_symbol: Dict[str, pd.DataFrame], window: int = 20) -> Tuple[np.ndarray, List[str]]:
    """Construct a simple feature matrix for each symbol.

    Features per symbol:
    - recent return (last / first - 1) over window
    - volatility (std of pct_change) over window
    - median volume over window
    - momentum (mean of last k returns)

    Returns X (n_symbols x n_features) and list of symbols in same order.
    """
    syms = []
    rows = []
    for s, df in (prices_by_symbol or {}).items():
        syms.append(s)
        if df is None or df.empty or 'close' not in df.columns:
            rows.append([0.0, 0.0, 0.0, 0.0])
            continue
        c = df['close'].astype(float).dropna()
        v = df.get('volume')
        windowed = c.iloc[-window:]
        if len(windowed) == 0:
            rows.append([0.0, 0.0, 0.0, 0.0])
            continue
        recent_return = float(windowed.iloc[-1] / windowed.iloc[0] - 1.0)
        rets = c.pct_change().dropna().iloc[-window:]
        vol = float(rets.std()) if not rets.empty else 0.0
        med_vol = float(v.astype(float).iloc[-window:].median()) if (v is not None and not v.dropna().empty) else 0.0
        momentum = float(rets.tail(3).mean()) if len(rets) >= 1 else 0.0
        rows.append([recent_return, vol, med_vol, momentum])
    X = np.asarray(rows, dtype=float)
    return X, syms


class OpportunityPredictor:
    """A thin wrapper around an sklearn regressor used to score symbols by opportunity.

    It supports `fit(X, y)`, `predict(X)` and `save(path)` / `load(path)`.
    """

    def __init__(self, model=None):
        if model is None and RandomForestRegressor is not None:
            self.model = Pipeline([( 'scale', StandardScaler()), ('rf', RandomForestRegressor(n_estimators=50, random_state=0) )])
        else:
            self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.model is None:
            raise RuntimeError('scikit-learn is required for training OpportunityPredictor')
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            # fallback: simple heuristic score (recent_return * 100 - vol * 10 + momentum)
            return (X[:, 0] * 100.0) - (X[:, 1] * 10.0) + (X[:, 3] * 100.0)
        return self.model.predict(X)

    def save(self, path: str):
        if self.model is None:
            # save minimal JSON of heuristic
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump({'heuristic': True}, fh)
            return
        # use pickle for full model
        import pickle

        with open(path, 'wb') as fh:
            pickle.dump(self.model, fh)

    @classmethod
    def load(cls, path: str) -> 'OpportunityPredictor':
        import os

        if not os.path.exists(path):
            raise FileNotFoundError(path)
        try:
            import pickle

            with open(path, 'rb') as fh:
                model = pickle.load(fh)
            return cls(model=model)
        except Exception:
            # fallback to JSON heuristic loader
            with open(path, 'r', encoding='utf-8') as fh:
                _ = json.load(fh)
            return cls(model=None)


class AnomalyDetector:
    """Thin wrapper for IsolationForest to flag anomalous symbols based on features."""

    def __init__(self, model=None):
        if model is None and IsolationForest is not None:
            self.model = IsolationForest(random_state=0, n_estimators=50)
        else:
            self.model = model

    def fit(self, X: np.ndarray):
        if self.model is None:
            raise RuntimeError('scikit-learn is required for AnomalyDetector')
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            # no model: return all non-anomalous
            return np.ones(X.shape[0], dtype=int)
        pred = self.model.predict(X)
        # sklearn's IsolationForest returns 1 for normal, -1 for anomalous
        return (pred == 1).astype(int)


def ml_rank_universe(prices_by_symbol: Dict[str, pd.DataFrame], predictor: Optional[OpportunityPredictor] = None, anomaly_detector: Optional[AnomalyDetector] = None, window: int = 20) -> List[Tuple[str, float, bool]]:
    """Rank symbols using ML predictor; returns list of tuples (symbol, score, is_normal).

    If `predictor` is None, uses heuristic scoring embedded in `OpportunityPredictor.predict`.
    """
    X, syms = build_basic_features(prices_by_symbol, window=window)
    pred = (predictor.predict(X) if predictor is not None else OpportunityPredictor().predict(X))
    is_normal = (anomaly_detector.predict(X) if anomaly_detector is not None else np.ones(len(syms), dtype=int))
    # pair symbol with predicted score and normal flag
    out = list(zip(syms, pred.tolist(), [bool(int(v)) for v in is_normal]))
    # sort by descending score (higher opportunity first)
    out = sorted(out, key=lambda t: t[1], reverse=True)
    return out


__all__ = [
    'build_basic_features',
    'OpportunityPredictor',
    'AnomalyDetector',
    'ml_rank_universe',
]
