from __future__ import annotations

from typing import Dict, Any
import pandas as pd


class Momentum:
    """Simple momentum strategy using N-period price difference.

    Parameters
    ----------
    window: int
        Lookback for momentum (periods)
    size: float
        Position size
    threshold: float
        Minimum normalized momentum (fractional change) to trigger
    """

    def __init__(self, window: int = 5, size: float = 1.0, threshold: float = 0.0):
        self.window = int(window)
        self.size = float(size)
        self.threshold = float(threshold)

    def generate_targets(self, prices: pd.DataFrame) -> pd.Series:
        df = prices.copy()
        if "close" not in df.columns:
            raise ValueError("prices must contain 'close' column")
        close = df["close"].astype(float)
        mom = close.diff(self.window)
        # normalized fractional change
        frac = mom / close.shift(self.window).replace(0.0, pd.NA)
        signal = pd.Series(0.0, index=df.index)
        signal[frac.fillna(0.0) > self.threshold] = 1.0
        signal[frac.fillna(0.0) < -self.threshold] = -1.0
        positions = signal * self.size
        positions.index = df.index
        return positions.rename("target")
