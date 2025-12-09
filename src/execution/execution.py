from __future__ import annotations

from typing import List, Tuple
import pandas as pd
import numpy as np
import datetime as dt


def twap_schedule(start: pd.Timestamp, end: pd.Timestamp, total_qty: float, intervals: int) -> List[Tuple[pd.Timestamp, float]]:
    """Return a TWAP schedule: list of (timestamp, qty) evenly spaced between start and end.

    - `start` and `end` must be pandas.Timestamp or coercible.
    - `intervals` number of slices (integer >=1).
    """
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    if intervals < 1:
        raise ValueError('intervals must be >=1')
    total_qty = float(total_qty)
    times = pd.date_range(start=start, end=end, periods=intervals)
    per = total_qty / intervals
    return [(t, per) for t in times]


def vwap_allocation(trades: pd.DataFrame, total_qty: float) -> pd.DataFrame:
    """Allocate `total_qty` across `trades` by historical volume proportion to approximate VWAP execution.

    Expects `trades` DataFrame with columns ['timestamp','price','volume'] or index aligned.
    Returns a DataFrame with an added `alloc_qty` column.
    """
    df = trades.copy()
    if 'volume' not in df.columns:
        raise ValueError('trades must contain volume column')
    vol = df['volume'].fillna(0.0)
    total_vol = vol.sum() if vol.sum() > 0 else 1.0
    df['alloc_qty'] = (vol / total_vol) * float(total_qty)
    return df


def simple_slippage(qty: float, price: float, impact_per_unit: float = 1e-6) -> float:
    """Estimate slippage (price units) as linear impact: impact_per_unit * qty.

    This is a placeholder estimator and intentionally simple.
    """
    return float(impact_per_unit) * float(qty)


__all__ = ['twap_schedule', 'vwap_allocation', 'simple_slippage']
