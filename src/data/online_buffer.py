from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd
import datetime

ROOT = Path(__file__).resolve().parents[2]
BUFFER_DIR = ROOT / "data" / "online_buffer"


def ensure_buffer_dir() -> None:
    BUFFER_DIR.mkdir(parents=True, exist_ok=True)


def append_rows(df: pd.DataFrame, date: Optional[datetime.date] = None) -> Path:
    """Append rows to today's buffer parquet file.

    The buffer is partitioned by date under `data/online_buffer/{YYYYMMDD}.parquet`.
    Returns the path written.
    """
    ensure_buffer_dir()
    if date is None:
        date = pd.Timestamp.utcnow().date()
    path = BUFFER_DIR / f"{date.strftime('%Y%m%d')}.parquet"
    if path.exists():
        # append by reading and concatenating small dfs to preserve schema
        existing = pd.read_parquet(path)
        out = pd.concat([existing, df], ignore_index=True)
    else:
        out = df
    out.to_parquet(path, index=False)
    return path


def sample_recent(max_rows: int = 2000, lookback_days: int = 2) -> pd.DataFrame:
    """Read the most recent buffer parquet files (up to lookback_days) and return the tail.

    This is intentionally simple; in production you'd stream or use a proper feature store.
    """
    ensure_buffer_dir()
    files = sorted(BUFFER_DIR.glob('*.parquet'))
    if not files:
        return pd.DataFrame()
    # take recent files up to lookback_days
    cutoff = pd.Timestamp.utcnow().date() - pd.Timedelta(days=lookback_days)
    recent = [p for p in files if pd.to_datetime(p.stem, format='%Y%m%d', errors='coerce').date() >= cutoff]
    if not recent:
        recent = files[-1:]
    df = pd.concat([pd.read_parquet(p) for p in recent], ignore_index=True)
    if len(df) > max_rows:
        return df.tail(max_rows)
    return df
