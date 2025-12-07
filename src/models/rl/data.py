"""Helpers to load historical minute parquet into DataFrames suitable for ReplayEnv.

This loader expects minute parquet files stored under:
  <data_root>/<pair>/<YYYYMMDD>/<YYYYMMDDTHHMM>.parquet

Files are expected to contain at least 'timestamp' (datetime) and either
'vwap' or 'close' columns. The loader will produce a DataFrame with
columns ['timestamp', 'close'] sorted by timestamp and without duplicates.
"""
from __future__ import annotations

import os
from typing import Optional
import pandas as pd


def load_price_history(data_root: str, pair: str, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Load minute price parquet files for `pair` from `data_root`.

    Parameters
    ----------
    data_root: str
        Root folder containing per-pair folders (e.g., 'data/raw')
    pair: str
        Pair folder name (e.g., 'XBT/USD')
    start, end: optional pd.Timestamp to filter the returned history

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['timestamp', 'close'] sorted by timestamp
    """
    pair_dir = os.path.join(data_root, pair)
    if not os.path.exists(pair_dir):
        return pd.DataFrame(columns=["timestamp", "close"])

    parts = []
    for root, dirs, files in os.walk(pair_dir):
        for f in files:
            if not f.endswith('.parquet'):
                continue
            p = os.path.join(root, f)
            try:
                df = pd.read_parquet(p)
            except Exception:
                # skip unreadable files
                continue
            # normalize timestamp column
            if "timestamp" not in df.columns and "time" in df.columns:
                df = df.rename(columns={"time": "timestamp"})
            if "timestamp" not in df.columns:
                continue
            # convert timestamp to pandas datetime (UTC if tz-aware)
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            # pick closing price from vwap if available, otherwise 'close'
            if "vwap" in df.columns:
                df["close"] = df["vwap"].astype(float)
            elif "close" in df.columns:
                df["close"] = df["close"].astype(float)
            else:
                continue
            parts.append(df[["timestamp", "close"]])

    if not parts:
        return pd.DataFrame(columns=["timestamp", "close"])

    res = pd.concat(parts, ignore_index=True)
    # drop duplicates and sort
    res = res.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if start is not None:
        res = res[res["timestamp"] >= pd.to_datetime(start, utc=True)]
    if end is not None:
        res = res[res["timestamp"] <= pd.to_datetime(end, utc=True)]

    return res
