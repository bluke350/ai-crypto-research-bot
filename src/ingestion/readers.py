from __future__ import annotations
import os
from typing import Optional
import hashlib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime


EXPECTED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _validate_df(df: pd.DataFrame):
    if df is None or df.empty:
        raise ValueError("DataFrame is empty")
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    # ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # sort
    df = df.sort_values("timestamp")
    return df


def snapshot_to_parquet(symbol: str, start: str, end: str, df: Optional[pd.DataFrame], out_root: str = "data/raw") -> str:
    """Write a snapshot to data/raw/{symbol}/{YYYYMMDD}/bars.parquet and return the file path.

    Validates schema and writes a companion .sha256 file with the hex digest.
    """
    if df is None:
        raise ValueError("df is required")
    df = _validate_df(df)
    # partition path by date from start
    date_str = datetime.strptime(start, "%Y-%m-%d").strftime("%Y%m%d")
    out_dir = os.path.join(out_root, symbol, date_str)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bars.parquet")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_path)
    # compute sha256
    h = hashlib.sha256()
    with open(out_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    sha_path = out_path + ".sha256"
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(h.hexdigest())
    return out_path
