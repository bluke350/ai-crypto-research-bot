from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pandas as pd


def load_checkpoint(path: str) -> dict:
    # Try torch.load first for PyTorch checkpoints (handles persistent IDs),
    # fall back to pickle.load for legacy pickled checkpoints.
    try:
        import torch

        try:
            return torch.load(path, map_location="cpu")
        except Exception:
            # fall through to pickle
            pass
    except Exception:
        # torch not available or failed to import; fallback to pickle
        pass
    with open(path, "rb") as fh:
        return pickle.load(fh)


class ModelWrapper:
    """Wraps a standardized checkpoint and provides prediction utilities.

    The checkpoint is expected to follow the format produced by `train_ml`:
    {
      'type': 'linear_baseline',
      'seed': ...,
      'steps': ...,
      'model': { 'sklearn_model': <model> } OR { 'coef': array },
      'features': { 'n_features': 2, 'lagged': [1,2] },
      'metrics': {...}
    }
    """

    def __init__(self, checkpoint: dict[str, Any]):
        self.ckpt = checkpoint
        self.model_info = checkpoint.get("model", {})
        # support RL-style torch checkpoints saved with keys 'actor'/'critic'
        self._rl_actor = None
        actor_state = None
        if isinstance(checkpoint, dict):
            # top-level actor (PyTorch state_dict) e.g. produced by PPOTrainer.save
            actor_state = checkpoint.get("actor")
            # some checkpoints may nest model under 'model' key
            if actor_state is None and isinstance(self.model_info, dict):
                actor_state = self.model_info.get("actor")
        if actor_state is not None:
            # convert tensors to numpy arrays where needed and extract MLP layers
            try:
                import torch

                def _to_np(v):
                    return v.detach().cpu().numpy() if hasattr(v, "detach") else (v.numpy() if hasattr(v, "numpy") else v)
            except Exception:
                def _to_np(v):
                    return v

            # collect ordered net.*.weight and net.*.bias entries
            layers = []
            # keys like 'net.0.weight', 'net.0.bias', 'net.2.weight', ...
            net_items = {}
            for k, v in actor_state.items():
                if k.startswith("net.") and (k.endswith("weight") or k.endswith("bias")):
                    net_items[k] = _to_np(v)
            # group by layer index
            idxs = sorted({int(k.split(".")[1]) for k in net_items.keys()})
            for i in idxs:
                w_key = f"net.{i}.weight"
                b_key = f"net.{i}.bias"
                if w_key in net_items and b_key in net_items:
                    W = net_items[w_key]
                    b = net_items[b_key]
                    layers.append((W, b))
            # log_std if present
            log_std = actor_state.get("log_std")
            if log_std is not None:
                try:
                    log_std = _to_np(log_std)
                except Exception:
                    pass

            if layers:
                self._rl_actor = {"layers": layers, "log_std": log_std}

    @classmethod
    def from_file(cls, path: str):
        return cls(load_checkpoint(path))

    def predict_returns(self, prices: pd.DataFrame) -> pd.Series:
        """Predict next-step returns for provided `prices` DataFrame.

        Returns a Series aligned to prices.index[2:] (because we use lag1 and lag2).
        """
        if "close" not in prices.columns:
            raise ValueError("prices DataFrame must contain 'close' column")
        close = prices["close"].astype(float).values
        ret = np.concatenate([[0.0], np.diff(close) / close[:-1]])
        lag1 = np.roll(ret, 1)
        lag2 = np.roll(ret, 2)
        X = np.column_stack([lag1, lag2])[2:]
        # predict
        if "sklearn_model" in self.model_info:
            model = self.model_info["sklearn_model"]
            y_pred = model.predict(X)
        elif self._rl_actor is not None:
            # use RL actor MLP to produce action mean for each time step
            layers = self._rl_actor["layers"]

            def _mlp_forward(x_row):
                a = x_row.astype(float)
                # numpy forward: for each (W,b) apply tanh for hidden layers, linear for last
                for i, (W, b) in enumerate(layers):
                    a = a.dot(W.T) + b
                    # apply tanh on all but last
                    if i < len(layers) - 1:
                        a = np.tanh(a)
                return a

            # build observations for RL: use last N returns where N = input dim
            in_dim = layers[0][0].shape[1]
            rets_full = ret  # earlier computed ret array (length == len(close))
            obs_rows = []
            for i in range(2, len(close)):
                # build observation as last `in_dim` returns ending at index i
                start = max(0, i - in_dim + 1)
                window = rets_full[start : i + 1]
                if len(window) < in_dim:
                    # pad with zeros at left
                    pad = np.zeros(in_dim - len(window))
                    obs = np.concatenate([pad, window])
                else:
                    obs = window
                obs_rows.append(obs)
            if not obs_rows:
                y_pred = np.zeros((0,))
            else:
                Y = np.vstack(obs_rows)
                pred = np.apply_along_axis(_mlp_forward, 1, Y)
                # pred may be multi-dim; collapse to scalar by taking first component
                if pred.ndim == 1:
                    y_pred = pred
                else:
                    y_pred = pred[:, 0]
        else:
            raw = self.model_info.get("coef")
            if raw is None:
                raise ValueError("Model checkpoint missing 'coef' in 'model' section")
            coef = np.asarray(raw)
            coef = coef.reshape(-1)
            # handle several coef shapes robustly:
            # - exact match: [b1, b2] -> X.dot(coef)
            # - intercept + coefs: [intercept, b1, b2] -> X.dot(b)+intercept
            # - scalar coef: broadcast to all features
            # - otherwise: raise informative error
            if coef.size == X.shape[1]:
                y_pred = X.dot(coef)
            elif coef.size == X.shape[1] + 1:
                intercept = float(coef[0])
                beta = coef[1:]
                y_pred = X.dot(beta) + intercept
            elif coef.size == 1:
                # broadcast scalar coefficient across features
                y_pred = X.dot(np.repeat(float(coef.item()), X.shape[1]))
            else:
                raise ValueError(
                    f"Coefficient size ({coef.size}) incompatible with feature dim ({X.shape[1]})."
                )

        idx = prices.index[2:]
        return pd.Series(y_pred, index=idx, name="predicted_return")

    def predicted_to_targets(self, prices: pd.DataFrame, method: str = "threshold", *,
                             size: float = 1.0, threshold: float = 0.0005, vol_window: int = 20,
                             scale: float = 1.0, max_size: float | None = None) -> pd.Series:
        """Map predicted returns to position target units.

        Methods:
        - "threshold": simple threshold-based sizing (long if pred > threshold,
          short if pred < -threshold) with fixed `size`.
        - "vol_norm": volatility-normalized sizing. Computes rolling std of
          percent returns from `prices` using `vol_window`, then produces
          position = scale * pred / vol, clipped to [-max_size, max_size].

        Returns a Series aligned to `prices.index[2:]` (same as predictions).
        """
        preds = self.predict_returns(prices)
        if method == "threshold":
            sig = preds.copy()
            sig.loc[preds.abs() <= threshold] = 0.0
            sig.loc[preds > threshold] = float(size)
            sig.loc[preds < -threshold] = float(-size)
            return sig.rename("target")
        elif method == "vol_norm":
            # compute returns from prices and rolling volatility aligned to preds
            if "close" not in prices.columns:
                raise ValueError("prices DataFrame must contain 'close' column")
            close = prices["close"].astype(float)
            ret = np.concatenate([[0.0], np.diff(close) / close[:-1]])
            vol = pd.Series(ret).rolling(vol_window, min_periods=1).std(ddof=0).values
            # align vol to preds index (preds start at index 2)
            vol_preds = np.asarray(vol[2:], dtype=float)
            eps = 1e-12
            raw = scale * (np.asarray(preds.values, dtype=float) / (vol_preds + eps))
            if max_size is not None:
                raw = np.clip(raw, -abs(max_size), abs(max_size))
            series = pd.Series(raw, index=preds.index, name="target")
            return series
        else:
            raise ValueError(f"unknown method: {method}")
