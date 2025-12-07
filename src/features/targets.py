from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple


def triple_barrier_labels(close: pd.Series, pt: float = 0.01, sl: float = 0.005, horizon: int = 60) -> pd.Series:
    """Return triple-barrier labels for a close price series.

    Labels: 1 for profit-taking first, -1 for stop-loss first, 0 for horizon/no-touch.
    """
    if close.empty:
        return pd.Series(dtype=int)
    n = len(close)
    out = pd.Series(0, index=close.index)
    for i in range(n):
        base = close.iat[i]
        if pd.isna(base):
            out.iat[i] = 0
            continue
        end = min(n, i + horizon + 1)
        window = close.iloc[i + 1 : end]
        up = base * (1 + pt)
        down = base * (1 - sl)
        touched_up = window[window >= up]
        touched_down = window[window <= down]
        if not touched_up.empty and not touched_down.empty:
            # which happened first
            if touched_up.index[0] < touched_down.index[0]:
                out.iat[i] = 1
            else:
                out.iat[i] = -1
        elif not touched_up.empty:
            out.iat[i] = 1
        elif not touched_down.empty:
            out.iat[i] = -1
        else:
            out.iat[i] = 0
    return out
