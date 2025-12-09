from __future__ import annotations

import os
import pickle
from typing import Optional

import glob
import numpy as np
import pandas as pd
import json


def _load_csvs(data_root: str):
    paths = glob.glob(os.path.join(data_root, "*.csv")) if data_root else []
    frames = []
    for p in paths:
        try:
            from src.utils.io import load_prices_csv
            df = load_prices_csv(p, dedupe='first')
            frames.append(df)
        except Exception:
            continue
    if frames:
        return pd.concat(frames, ignore_index=True)
    return None


def _fit_linear_regression(X: np.ndarray, y: np.ndarray):
    try:
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(X, y)
        return {"sklearn_model": model}
    except Exception:
        # fallback to numpy least squares
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return {"coef": coef}


def train_ml(data_root: str = "data/raw", save: str = "models/ml_baseline.pkl", steps: int = 200, seed: int = 0,
             val_split: float = 0.2, patience: int = 10, save_best: bool = True,
             data_df: Optional[pd.DataFrame] = None) -> str:
    """Train a tiny linear-regression baseline to predict next-step returns.

    Behavior:
    - If CSV files exist in `data_root`, load them and construct features.
    - Otherwise, synthesize a small dataset.
    - Train a linear model (sklearn if available, otherwise numpy lstsq) and
      persist the model dict as a pickle file.
    """
    os.makedirs(os.path.dirname(save) or ".", exist_ok=True)

    # if caller provided a DataFrame (balanced/sample), use it directly
    if data_df is not None:
        df = data_df.copy()
    else:
        df = _load_csvs(data_root) if data_root else None
    if df is None or df.empty:
        # synthesize data: simple AR(1) returns
        rng = np.random.default_rng(int(seed))
        n = max(100, int(steps))
        returns = rng.normal(scale=0.001, size=n)
        price = 100.0 + np.cumsum(returns)
        df = pd.DataFrame({"close": price})

    # build features: use lag-1, lag-2 returns
    close = df["close"].astype(float).values
    ret = np.concatenate([[0.0], np.diff(close) / close[:-1]])
    # create lagged features
    lag1 = np.roll(ret, 1)
    lag2 = np.roll(ret, 2)
    X = np.column_stack([lag1, lag2])[2:]
    y = ret[2:]
    # split into train/validation sets (preserve order but do a simple index split)
    n = X.shape[0]
    idx = np.arange(n)
    split = int(n * (1.0 - float(val_split)))
    train_idx = idx[:split]
    val_idx = idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    best = None
    best_mse = float("inf")
    no_improve = 0

    # iterate multiple bootstrap fits and keep best on validation set
    for step in range(int(steps)):
        # create a bootstrap sample from training data to introduce stochasticity
        if len(X_train) > 0:
            rng = np.random.default_rng(int(seed) + step)
            bs_idx = rng.integers(0, len(X_train), size=len(X_train))
            X_bs = X_train[bs_idx]
            y_bs = y_train[bs_idx]
        else:
            X_bs, y_bs = X_train, y_train

        model_info = _fit_linear_regression(X_bs, y_bs)

        # evaluate on validation set
        if "sklearn_model" in model_info:
            sklearn_model = model_info["sklearn_model"]
            y_pred_val = sklearn_model.predict(X_val) if len(X_val) else np.array([])
        else:
            coef = np.asarray(model_info.get("coef"))
            coef = coef.reshape(-1)
            y_pred_val = X_val.dot(coef) if len(X_val) else np.array([])

        mse = float(np.mean((y_val - y_pred_val) ** 2)) if len(y_val) else float(np.mean((y - (X.dot(coef) if "coef" in locals() else y_pred_val)) ** 2))
        # if improved, save
        if mse < best_mse:
            best_mse = mse
            best = {
                "type": "linear_baseline",
                "seed": int(seed),
                "steps": int(steps),
                "model": model_info,
                "features": {"n_features": X.shape[1], "lagged": [1, 2]},
                "metrics": {"val_mse": mse},
            }
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= int(patience):
            break

    if best is None:
        # fallback to training on full data
        model_info = _fit_linear_regression(X, y)
        best = {
            "type": "linear_baseline",
            "seed": int(seed),
            "steps": int(steps),
            "model": model_info,
            "features": {"n_features": X.shape[1], "lagged": [1, 2]},
        }

    # compute simple metrics on full dataset for bookkeeping
    if "sklearn_model" in best["model"]:
        m = best["model"]["sklearn_model"]
        y_pred = m.predict(X)
    else:
        coef = np.asarray(best["model"].get("coef"))
        coef = coef.reshape(-1)
        y_pred = X.dot(coef)

    mse = float(np.mean((y - y_pred) ** 2))
    mae = float(np.mean(np.abs(y - y_pred)))
    best["metrics"]["mse"] = mse
    best["metrics"]["mae"] = mae

    # Persist checkpoint, metrics, and metadata
    os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
    with open(save, "wb") as fh:
        pickle.dump(best, fh)

    metrics_path = os.path.splitext(save)[0] + ".metrics.json"
    try:
        with open(metrics_path, "w", encoding="utf-8") as mf:
            json.dump(best.get("metrics", {}), mf)
    except Exception:
        pass

    # metadata file with training params and a small provenance block
    meta = {
        "checkpoint": os.path.abspath(save),
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "train": {"data_root": data_root, "val_split": val_split, "steps": int(steps), "seed": int(seed), "patience": int(patience)},
        "features": best.get("features", {}),
        "metrics": best.get("metrics", {}),
    }
    meta_path = os.path.splitext(save)[0] + ".meta.json"
    try:
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta, mf)
    except Exception:
        pass

    # Save validation predictions and labels next to the checkpoint to support ensemble weight fitting
    try:
        import numpy as _np
        # If X_val/y_val are defined above, use them; else skip
        if 'X_val' in locals() and len(X_val) > 0:
            if 'sklearn_model' in best['model']:
                preds_val = best['model']['sklearn_model'].predict(X_val)
            else:
                coef = _np.asarray(best['model'].get('coef'))
                coef = coef.reshape(-1)
                preds_val = X_val.dot(coef)
            val_labels = y_val
            base = os.path.splitext(save)[0]
            preds_path = f"{base}.preds.npy"
            labels_path = f"{base}.val_labels.npy"
            _np.save(preds_path, _np.asarray(preds_val))
            _np.save(labels_path, _np.asarray(val_labels))
    except Exception:
        pass

    return save
