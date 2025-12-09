from __future__ import annotations

from typing import List
import pandas as pd


def resample_ohlcv(df: pd.DataFrame, rule: str = '5T') -> pd.DataFrame:
    """Resample OHLCV DataFrame indexed by datetime to a new timeframe `rule` (pandas offset alias).

    Expects columns: ['open','high','low','close','volume'] and a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError('DataFrame must have a DatetimeIndex')
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    res = df.resample(rule).agg(agg).dropna(how='all')
    return res


def multitimeframe_fusion(base: pd.DataFrame, higher_frames: List[pd.DataFrame], how: str = 'left') -> pd.DataFrame:
    """Fuse base timeframe features with a list of higher timeframe feature frames.

    Returns merged DataFrame aligned to `base` index. Each higher frame should have a DatetimeIndex and features.
    The fusion strategy is to forward-fill the higher timeframe features onto base timestamps.
    """
    merged = base.copy()
    for i, hf in enumerate(higher_frames):
        # prefix columns to avoid collisions
        hf_pref = hf.copy()
        hf_pref.columns = [f"hf{i}_{c}" for c in hf_pref.columns]
        # reindex to base using forward fill
        hf_aligned = hf_pref.reindex(merged.index, method='ffill')
        merged = merged.merge(hf_aligned, left_index=True, right_index=True, how=how)
    return merged


__all__ = ['resample_ohlcv', 'multitimeframe_fusion']
