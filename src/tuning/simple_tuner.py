from __future__ import annotations

import random
from typing import Dict, Any, Callable, Optional, Tuple


class RandomSearchTuner:
    """Very small random search tuner for local experiments.

    param_space: mapping from name -> list of possible values (discrete)
    objective: callable(params) -> float (lower is better)
    """

    def __init__(self, param_space: Dict[str, list], n_trials: int = 20, seed: Optional[int] = None):
        self.param_space = param_space
        self.n_trials = int(n_trials)
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def _sample(self) -> Dict[str, Any]:
        out = {}
        for k, vals in self.param_space.items():
            if isinstance(vals, list):
                out[k] = random.choice(vals)
            else:
                # if not a list, assume it's an iterator or callable
                try:
                    out[k] = vals()
                except Exception:
                    out[k] = vals
        return out

    def tune(self, objective: Callable[[Dict[str, Any]], float]) -> Tuple[Dict[str, Any], float]:
        best_params = None
        best_score = float("inf")
        for _ in range(self.n_trials):
            params = self._sample()
            try:
                score = float(objective(params))
            except Exception:
                score = float("inf")
            if score < best_score:
                best_score = score
                best_params = params
        return best_params or {}, best_score
