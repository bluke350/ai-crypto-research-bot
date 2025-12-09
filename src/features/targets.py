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


def triple_barrier_vectorized(close: pd.Series, pt: float = 0.01, sl: float = 0.005, horizon: int = 60) -> pd.Series:
    """Vectorized triple barrier labeling. Returns 1, -1, or 0.

    This implementation still uses a forward scan per-row but avoids heavy
    Python-level indexing inside the inner loop by using pandas slicing.
    It is not fully O(1) per-row but more efficient than naive indexing for
    moderate-size series.
    """
    if close.empty:
        return pd.Series(dtype=int)
    n = len(close)
    out = pd.Series(0, index=close.index)
    close_vals = close.values
    for i in range(n):
        base = close_vals[i]
        if pd.isna(base):
            out.iat[i] = 0
            continue
        end = min(n, i + horizon + 1)
        window = close_vals[i + 1 : end]
        if window.size == 0:
            out.iat[i] = 0
            continue
        up = base * (1 + pt)
        down = base * (1 - sl)
        # compute boolean arrays
        touched_up = window >= up
        touched_down = window <= down
        if touched_up.any() and touched_down.any():
            # which happened first
            up_idx = int(np.argmax(touched_up))
            dn_idx = int(np.argmax(touched_down))
            out.iat[i] = 1 if up_idx < dn_idx else -1
        elif touched_up.any():
            out.iat[i] = 1
        elif touched_down.any():
            out.iat[i] = -1
        else:
            out.iat[i] = 0
    return out


def meta_labels(close: pd.Series, pt: float = 0.01, sl: float = 0.005, horizon: int = 60) -> pd.DataFrame:
    """Return labels and time-to-touch/hit information as DataFrame.

    Columns: `label` (1/-1/0), `ttt` (time-to-touch in bars, NaN if none), `horizon` (bars)
    """
    labels = triple_barrier_vectorized(close, pt=pt, sl=sl, horizon=horizon)
    n = len(close)
    ttt = pd.Series(index=close.index, dtype=float)
    close_vals = close.values
    for i in range(n):
        if labels.iat[i] == 0:
            ttt.iat[i] = float('nan')
            continue
        base = close_vals[i]
        up = base * (1 + pt)
        down = base * (1 - sl)
        end = min(n, i + horizon + 1)
        window = close_vals[i + 1 : end]
        if labels.iat[i] == 1:
            hits = window >= up
        else:
            hits = window <= down
        if hits.any():
            ttt.iat[i] = float(int(np.argmax(hits)) + 1)
        else:
            ttt.iat[i] = float('nan')
    return pd.DataFrame({"label": labels, "ttt": ttt, "horizon": pd.Series([horizon] * n, index=close.index)})
