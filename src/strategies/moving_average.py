from __future__ import annotations

from typing import Dict, Any
import pandas as pd


class MovingAverageCrossover:
    """Simple moving-average crossover strategy that produces target positions.

    Parameters
    ----------
    short: int
        Short window length in periods
    long: int
        Long window length in periods
    size: float
        Position size when signal is active (units)
    """

    def __init__(self, short: int = 5, long: int = 20, size: float = 1.0):
        if short >= long:
            raise ValueError("short must be < long")
        self.short = int(short)
        self.long = int(long)
        self.size = float(size)

    def generate_targets(self, prices: pd.DataFrame) -> pd.Series:
        """Return a Series of target position units aligned to prices index.

        Expects `prices` DataFrame to contain a 'close' column and a 'timestamp' index or column.
        """
        df = prices.copy()
        # ensure close exists
        if "close" not in df.columns:
            raise ValueError("prices must contain 'close' column")
        close = df["close"].astype(float)
        short_ma = close.rolling(self.short, min_periods=1).mean()
        long_ma = close.rolling(self.long, min_periods=1).mean()
        signal = (short_ma > long_ma).astype(float)
        # map signal to positions: 1.0 for long, -1.0 for short/back to neutral
        positions = signal.replace({0.0: -1.0}) * self.size
        # keep the same index as the input prices DataFrame (so callers slicing by iloc still align)
        positions.index = df.index
        return positions.rename("target")
