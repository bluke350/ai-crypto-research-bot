from __future__ import annotations

from typing import Dict, Any
import pandas as pd


class VolatilityBreakout:
    """Volatility breakout strategy using rolling highs/lows and a threshold.

    Parameters
    ----------
    window: int
        Lookback window for highs/lows
    threshold: float
        Fractional threshold above high / below low to trigger breakout
    size: float
        Position size
    """

    def __init__(self, window: int = 20, threshold: float = 0.005, size: float = 1.0):
        self.window = int(window)
        self.threshold = float(threshold)
        self.size = float(size)

    def generate_targets(self, prices: pd.DataFrame) -> pd.Series:
        df = prices.copy()
        # allow use of 'high'/'low' if present, else use 'close'
        high = df["high"] if "high" in df.columns else df["close"]
        low = df["low"] if "low" in df.columns else df["close"]
        close = df["close"]
        # compare to prior-window high/low (shifted) so breakouts trigger
        rolling_high = high.rolling(self.window, min_periods=1).max().shift(1)
        rolling_low = low.rolling(self.window, min_periods=1).min().shift(1)
        long_signal = close > (rolling_high * (1.0 + self.threshold))
        short_signal = close < (rolling_low * (1.0 - self.threshold))
        signal = pd.Series(0.0, index=df.index)
        signal[long_signal] = 1.0
        signal[short_signal] = -1.0
        positions = signal * self.size
        positions.index = df.index
        return positions.rename("target")
