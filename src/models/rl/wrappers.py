"""Environment wrappers for RL envs used by the trainer/evaluator.

RelativeActionWrapper converts an action interpreted as a relative delta
into a desired absolute position by adding the delta to the env's current
position attribute.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


class RelativeActionWrapper:
    def __init__(self, env):
        self.env = env

    def reset(self):
        return self.env.reset()

    def _get_position(self) -> float:
        # try common attribute names
        for name in ("position", "pos", "position_size"):
            if hasattr(self.env, name):
                return float(getattr(self.env, name))
        # fallback
        return 0.0

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        # action is interpreted as delta; desired = current_position + delta
        delta = float(np.array(action).ravel()[0])
        current = self._get_position()
        desired = current + delta
        return self.env.step(np.array([desired], dtype=float))

    # expose other attributes for convenience
    def __getattr__(self, name):
        return getattr(self.env, name)


class NormalizedObservationWrapper:
    """Wrap an env and normalize the price observation.

    Modes:
      - 'raw': no change
      - 'returns': price_t -> (price_t - price_{t-1}) / price_{t-1}
      - 'logdiff': price_t -> log(price_t) - log(price_{t-1})

    The wrapper expects the wrapped env to expose price as the first element of
    the observation vector and to reset/step returning numpy arrays.
    """
    def __init__(self, env, mode: str = "returns"):
        self.env = env
        self.mode = mode
        self._last_price = None

    def reset(self):
        obs = self.env.reset()
        price = float(obs[0])
        self._last_price = price
        return self._normalize_obs(obs, initial=True)

    def _normalize_obs(self, obs, initial: bool = False):
        price = float(obs[0])
        pos = float(obs[1]) if len(obs) > 1 else 0.0
        if self.mode == "raw" or self._last_price is None or initial:
            norm_price = 0.0 if self.mode != "raw" else price
        elif self.mode == "returns":
            if self._last_price == 0:
                norm_price = 0.0
            else:
                norm_price = (price - self._last_price) / float(self._last_price)
        else:  # logdiff
            import math
            if self._last_price <= 0 or price <= 0:
                norm_price = 0.0
            else:
                norm_price = math.log(price) - math.log(self._last_price)

        # update last_price for next step
        self._last_price = price
        return np.array([float(norm_price), float(pos)], dtype=np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        norm_obs = self._normalize_obs(obs)
        return norm_obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)
