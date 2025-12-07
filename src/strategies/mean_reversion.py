from __future__ import annotations

from typing import Dict, Any
import pandas as pd


class MeanReversion:
    """Mean reversion strategy based on z-score of returns.

    Parameters
    ----------
    window: int
        Rolling window for mean/std
    threshold: float
        z-score threshold to open a position
    size: float
        Position size in units
    """

    def __init__(self, window: int = 20, threshold: float = 1.5, size: float = 1.0):
        self.window = int(window)
        self.threshold = float(threshold)
        self.size = float(size)

    def generate_targets(self, prices: pd.DataFrame) -> pd.Series:
        df = prices.copy()
        if "close" not in df.columns:
            raise ValueError("prices must contain 'close' column")
        close = df["close"].astype(float)
        ret = close.pct_change().fillna(0.0)
        mean = ret.rolling(self.window, min_periods=1).mean()
        std = ret.rolling(self.window, min_periods=1).std(ddof=0).fillna(0.0)
        z = (ret - mean) / (std.replace(0.0, 1.0))
        # short when z > threshold, long when z < -threshold
        signal = pd.Series(0.0, index=df.index)
        signal[z > self.threshold] = -1.0
        signal[z < -self.threshold] = 1.0
        positions = signal * self.size
        positions.index = df.index
        return positions.rename("target")
