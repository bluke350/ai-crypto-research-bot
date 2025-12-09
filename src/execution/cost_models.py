from __future__ import annotations
from typing import Optional
import math
import random
import numpy as np


class FeeModel:
    def __init__(self, maker_bps: float = 16.0, taker_bps: float = 26.0, fixed_fee_pct: Optional[float] = None):
        self.maker_bps = maker_bps
        self.taker_bps = taker_bps
        self.fixed_fee_pct = fixed_fee_pct

    def compute(self, notional: float, is_maker: bool = False) -> float:
        if self.fixed_fee_pct is not None:
            return abs(notional) * float(self.fixed_fee_pct)
        bps = self.maker_bps if is_maker else self.taker_bps
        return abs(notional) * (bps / 10000.0)


class SlippageModel:
    def __init__(self, k: float = 0.1, daily_vol: float = 0.02, fixed_slippage_pct: Optional[float] = None, stochastic_sigma: float = 0.0, stochastic_mu: float = 0.0, seed: Optional[int] = None):
        self.k = k
        self.daily_vol = daily_vol
        self.fixed_slippage_pct = fixed_slippage_pct
        self.stochastic_sigma = stochastic_sigma
        self.stochastic_mu = stochastic_mu
        # Use an instance-local random generator for deterministic draws when seeded
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

    def estimate_pct(self, notional: float) -> float:
        if self.fixed_slippage_pct is not None:
            base = float(self.fixed_slippage_pct)
        else:
            if notional == 0:
                base = 0.0
            else:
                base = self.k * math.sqrt(abs(notional)) * self.daily_vol
        if self.stochastic_sigma and self.stochastic_sigma > 0.0:
            draw = float(self._rng.lognormal(mean=self.stochastic_mu, sigma=self.stochastic_sigma))
            return base * draw
        return base


class LatencySampler:
    def __init__(self, base_ms: int = 50, jitter_ms: int = 100, seed: Optional[int] = None):
        self.base = base_ms
        self.jitter = jitter_ms
        if seed is not None:
            random.seed(seed)

    def sample_ms(self) -> float:
        return float(self.base + random.random() * self.jitter)
