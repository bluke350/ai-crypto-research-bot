from __future__ import annotations

from typing import Tuple, Dict
import numpy as np
import pandas as pd


def volume_profile(price_series: pd.Series, volume_series: pd.Series, bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a volume profile: returns price bin edges and volume per bin.

    price_series and volume_series must be aligned Series of equal length.
    """
    if len(price_series) != len(volume_series):
        df = pd.concat([price_series, volume_series], axis=1, join='inner')
        df.columns = ['price', 'volume']
    else:
        df = pd.DataFrame({'price': price_series, 'volume': volume_series})
    prices = df['price'].values
    vols = df['volume'].values
    if len(prices) == 0:
        return np.array([]), np.array([])
    min_p = float(np.nanmin(prices))
    max_p = float(np.nanmax(prices))
    if max_p <= min_p:
        edges = np.linspace(min_p - 0.5, max_p + 0.5, bins + 1)
    else:
        edges = np.linspace(min_p, max_p, bins + 1)
    inds = np.searchsorted(edges, prices, side='right') - 1
    inds = np.clip(inds, 0, bins - 1)
    vol_per_bin = np.zeros(bins, dtype=float)
    for i, v in zip(inds, vols):
        vol_per_bin[i] += float(v)
    return edges, vol_per_bin


def vpoc(edges: np.ndarray, vol_per_bin: np.ndarray) -> float:
    """Return the price (approx) at the volume point of control (bin with max volume)."""
    if vol_per_bin.size == 0:
        return float('nan')
    idx = int(np.argmax(vol_per_bin))
    # return center of bin
    return float((edges[idx] + edges[idx + 1]) / 2.0)


def vpin(volume_series: pd.Series, window: int = 50, bucket_size: int = 50) -> pd.Series:
    """Compute a simple VPIN-like metric: imbalance of buy/sell volumes per bucket over rolling window.

    This is a simplified offline estimator that assumes trade sign by price movement.
    """
    # naive trade sign estimator: compare price changes
    # caller should pass price_series if available; for now expect volume_series index aligns with prices in caller
    # Here we implement a placeholder that returns rolling normalized variance of volume as a proxy
    vs = volume_series.fillna(0.0)
    if len(vs) < 2:
        return pd.Series([0.0] * len(vs), index=vs.index)
    rv = vs.rolling(window=min(window, len(vs))).std().fillna(0.0)
    # normalize
    denom = (rv.max() if rv.max() > 0 else 1.0)
    return (rv / denom).fillna(0.0)


__all__ = ["volume_profile", "vpoc", "vpin"]
