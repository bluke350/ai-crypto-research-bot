from __future__ import annotations

from typing import List, Tuple
import pandas as pd
import numpy as np


def detect_regimes(price_series: pd.Series, window: int = 252, thresholds: Tuple[float, float] = (0.33, 0.66)) -> pd.Series:
    """Detect simple volatility regimes from `price_series`.

    Returns a Series indexed like `price_series` with values in {'low','mid','high'}.
    - Computes log returns and rolling volatility (std) over `window` periods.
    - Computes global quantiles on the rolling vol series and assigns regimes.
    """
    if not isinstance(price_series.index, pd.DatetimeIndex):
        price_series = price_series.copy()
        try:
            price_series.index = pd.to_datetime(price_series.index)
        except Exception:
            # keep as-is if cannot coerce
            pass

    # compute log returns
    returns = np.log(price_series).diff()
    vol = returns.rolling(window=window, min_periods=1).std()
    # compute quantile thresholds on non-NaN values
    valid = vol.dropna()
    if valid.empty:
        # fallback to 'mid' for all
        return pd.Series(['mid'] * len(price_series), index=price_series.index)
    q_low = float(valid.quantile(thresholds[0]))
    q_high = float(valid.quantile(thresholds[1]))

    def map_vol(v: float) -> str:
        if pd.isna(v):
            return 'mid'
        if v <= q_low:
            return 'low'
        if v >= q_high:
            return 'high'
        return 'mid'

    regimes = vol.map(map_vol)
    # ensure same length/index as input
    regimes = regimes.reindex(price_series.index).fillna('mid')
    return regimes


def identify_regimes(prices: pd.Series, window: int = 60, high_vol_threshold: float = 0.02) -> pd.Series:
    """Identify simple volatility regimes: returns series -> rolling std; label high (1) vs low (0) volatility.

    Returns a Series of 0/1 labels aligned with input index.
    """
    if prices is None or prices.empty:
        return pd.Series(dtype=int)
    # compute simple returns
    rets = prices.pct_change().fillna(0.0)
    vol = rets.rolling(window=window, min_periods=1).std()
    labels = (vol >= high_vol_threshold).astype(int)
    labels.index = prices.index
    return labels


def regime_periods(labels: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Compress labels into contiguous periods: returns list of (start, end, label)."""
    out: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    if labels is None or labels.empty:
        return out
    prev = labels.iloc[0]
    start = labels.index[0]
    for idx, v in zip(labels.index[1:], labels.iloc[1:]):
        if v != prev:
            out.append((start, idx - pd.Timedelta(seconds=1), int(prev)))
            start = idx
            prev = v
    out.append((start, labels.index[-1], int(prev)))
    return out


__all__ = [
    "detect_regimes",
    "identify_regimes",
    "regime_periods",
]
