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

    def __init__(self, short: int = 5, long: int = 20, size: float = 1.0, vol_filter: bool = False, vol_window: int = 20, vol_threshold: float = 0.01):
        if short >= long:
            raise ValueError("short must be < long")
        self.short = int(short)
        self.long = int(long)
        self.size = float(size)
        # volatility filter: when enabled, suppress positions when rolling volatility exceeds threshold
        self.vol_filter = bool(vol_filter)
        self.vol_window = int(vol_window)
        self.vol_threshold = float(vol_threshold)

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

        # apply volatility filter if enabled: compute rolling std of pct_change
        if self.vol_filter:
            returns = close.pct_change().fillna(0.0)
            vol = returns.rolling(self.vol_window, min_periods=1).std()
            # when vol > threshold, set positions to 0 (no trading)
            mask = vol > self.vol_threshold
            positions = positions.where(~mask, other=0.0)
        # keep the same index as the input prices DataFrame (so callers slicing by iloc still align)
        positions.index = df.index
        return positions.rename("target")
