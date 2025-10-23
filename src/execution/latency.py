from __future__ import annotations
import random
from typing import Optional


class LatencyModel:
    def __init__(self, base_ms: int = 50, jitter_ms: int = 100, seed: Optional[int] = None):
        self.base = base_ms
        self.jitter = jitter_ms
        if seed is not None:
            random.seed(seed)

    def sample(self) -> float:
        """Return latency in milliseconds as a float."""
        return float(self.base + random.random() * self.jitter)
