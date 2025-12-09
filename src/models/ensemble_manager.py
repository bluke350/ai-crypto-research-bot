from __future__ import annotations

from typing import Dict, List, Sequence, Optional
import numpy as np


class EnsembleManager:
    """Simple ensemble manager that combines model predictions with dynamic weighting.

    Weights are stored per model and can be updated by passing recent performance metrics.
    Predictions are combined by weighted average; model names are keys in dicts.
    """

    def __init__(self, model_names: Sequence[str], initial_weights: Optional[Dict[str, float]] = None):
        self.model_names = list(model_names)
        if initial_weights is None:
            w = np.ones(len(self.model_names), dtype=float)
            w = w / w.sum()
            self.weights = {n: float(wi) for n, wi in zip(self.model_names, w)}
        else:
            total = sum(float(initial_weights.get(n, 0.0)) for n in self.model_names)
            if total <= 0:
                # fallback to uniform
                w = np.ones(len(self.model_names), dtype=float) / len(self.model_names)
                self.weights = {n: float(wi) for n, wi in zip(self.model_names, w)}
            else:
                self.weights = {n: float(initial_weights.get(n, 0.0)) / total for n in self.model_names}

    def predict(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine model predictions provided as a dict model_name -> numpy array.

        All prediction arrays must be broadcastable to the same shape; result is weighted average.
        """
        arrs = []
        ws = []
        for n in self.model_names:
            p = preds.get(n)
            if p is None:
                # treat missing model as zero prediction
                continue
            arrs.append(np.asarray(p, dtype=float))
            ws.append(self.weights.get(n, 0.0))
        if not arrs:
            raise ValueError('No predictions provided')
        # broadcast to common shape
        out = np.zeros_like(arrs[0], dtype=float)
        # normalize weights among provided models
        wsum = float(sum(ws)) if sum(ws) > 0 else 1.0
        for a, w in zip(arrs, ws):
            out = out + (a * (w / wsum))
        return out

    def update_weights(self, performance: Dict[str, float], decay: float = 0.9):
        """Update weights based on recent performance metrics (higher is better).

        A simple exponentially-weighted update: new_weight = decay * old + (1-decay) * normalized_perf
        Then re-normalize to sum to 1.
        """
        # performance: model_name -> score (e.g., recent Sharpe or return)
        perf_vals = [float(performance.get(n, 0.0)) for n in self.model_names]
        # normalize perf to positive values
        minv = min(perf_vals)
        shifted = [v - minv + 1e-8 for v in perf_vals]
        ssum = sum(shifted) if sum(shifted) > 0 else len(shifted)
        norm_perf = [v / ssum for v in shifted]
        # update
        for n, npf in zip(self.model_names, norm_perf):
            old = self.weights.get(n, 0.0)
            new = decay * old + (1.0 - decay) * float(npf)
            self.weights[n] = float(new)
        # renormalize
        total = sum(self.weights.values())
        if total > 0:
            for n in self.weights:
                self.weights[n] = float(self.weights[n] / total)

    def fit_weights_from_preds(self, preds: Dict[str, np.ndarray], y: np.ndarray, metric: str = 'mse') -> Dict[str, float]:
        """Fit ensemble weights from model predictions on a validation set.

        preds: mapping model_name -> numpy array (predictions aligned to y)
        y: ground-truth numpy array
        metric: 'mse' (lower better) or 'r2' (higher better)

        Returns the computed weights (and updates internal weights).
        """
        scores = {}
        y = np.asarray(y, dtype=float)
        for n in self.model_names:
            p = preds.get(n)
            if p is None:
                scores[n] = float('nan')
                continue
            p = np.asarray(p, dtype=float)
            try:
                if metric == 'mse':
                    val = float(np.mean((y - p) ** 2))
                    # lower is better -> convert to score by inverse
                    scores[n] = 1.0 / (val + 1e-12)
                elif metric == 'r2':
                    ss_res = float(np.sum((y - p) ** 2))
                    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
                    scores[n] = max(r2, 0.0)
                else:
                    scores[n] = float(np.mean((y - p) ** 2))
            except Exception:
                scores[n] = float('nan')

        # replace NaNs with small value
        for n in self.model_names:
            if not np.isfinite(scores.get(n, np.nan)):
                scores[n] = 1e-12

        # normalize to weights
        total = sum(scores.values())
        if total <= 0:
            # fallback to uniform
            w = np.ones(len(self.model_names), dtype=float) / len(self.model_names)
            self.weights = {n: float(wi) for n, wi in zip(self.model_names, w)}
        else:
            self.weights = {n: float(scores[n] / total) for n in self.model_names}

        return dict(self.weights)

    def get_weights(self) -> Dict[str, float]:
        return dict(self.weights)


__all__ = ['EnsembleManager']
