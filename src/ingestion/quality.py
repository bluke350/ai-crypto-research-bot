"""Ingestion data quality checks: missing-bar detection and simple anomaly/outlier flags.

Exports:
- detect_missing_bars(df, interval_seconds): returns dict with missing counts and max_gap_seconds
- detect_outliers(df, return_z=5.0, vol_z=5.0): returns counts of outlier rows
- compute_qa_metrics(df, interval_seconds, return_z, vol_z): aggregate metrics dict
"""
from __future__ import annotations

from datetime import timedelta
from typing import Dict, Any

import numpy as np
import pandas as pd


def _to_seconds(interval: str | int) -> int:
    if isinstance(interval, int):
        return int(interval)
    s = str(interval).strip()
    # expect patterns like '1m', '5s', '1h'
    if s.endswith('s'):
        return int(s[:-1])
    if s.endswith('m'):
        return int(s[:-1]) * 60
    if s.endswith('h'):
        return int(s[:-1]) * 3600
    # fallback: try parse as int seconds
    try:
        return int(s)
    except Exception:
        return 60


def detect_missing_bars(df: pd.DataFrame, interval: str | int = '1m') -> Dict[str, Any]:
    if df is None or df.empty:
        return {'missing_count': 0, 'gaps': [], 'max_gap_seconds': 0}
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp')
    secs = _to_seconds(interval)
    diffs = df['timestamp'].diff().dt.total_seconds().fillna(secs).astype(int)
    # missing where diff > expected
    gaps = (diffs > secs).astype(int)
    gap_intervals = []
    max_gap = 0
    missing_count = 0
    if gaps.any():
        idxs = diffs[diffs > secs].index
        for i in idxs:
            gap = int(diffs.loc[i])
            gap_intervals.append(gap)
            missing_count += int((gap // secs) - 1)
            if gap > max_gap:
                max_gap = gap
    return {
        'missing_count': int(missing_count),
        'gaps': gap_intervals,
        'max_gap_seconds': int(max_gap),
    }


def detect_outliers(df: pd.DataFrame, return_z: float = 5.0, vol_z: float = 5.0) -> Dict[str, Any]:
    if df is None or df.empty:
        return {'return_outliers': 0, 'vol_outliers': 0}
    # compute returns based on close
    df = df.copy()
    if 'close' in df.columns:
        df['ret'] = df['close'].pct_change().fillna(0)
    else:
        df['ret'] = 0.0
    if 'volume' not in df.columns:
        df['volume'] = 0.0
    # z-score
    ret_z = np.abs((df['ret'] - df['ret'].mean()) / (df['ret'].std() + 1e-12))
    vol_zs = np.abs((df['volume'] - df['volume'].mean()) / (df['volume'].std() + 1e-12))
    return_outliers = int((ret_z > return_z).sum())
    vol_outliers = int((vol_zs > vol_z).sum())
    return {'return_outliers': return_outliers, 'vol_outliers': vol_outliers}


def compute_qa_metrics(df: pd.DataFrame, interval: str | int = '1m', return_z: float = 5.0, vol_z: float = 5.0) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    metrics['num_rows'] = int(len(df)) if df is not None else 0
    if df is None or df.empty:
        metrics.update({'missing_count': 0, 'max_gap_seconds': 0, 'return_outliers': 0, 'vol_outliers': 0})
        return metrics
    miss = detect_missing_bars(df, interval)
    outs = detect_outliers(df, return_z=return_z, vol_z=vol_z)
    metrics.update({
        'start': str(df['timestamp'].min()),
        'end': str(df['timestamp'].max()),
        'missing_count': miss.get('missing_count', 0),
        'gaps': miss.get('gaps', []),
        'max_gap_seconds': miss.get('max_gap_seconds', 0),
        'return_outliers': outs.get('return_outliers', 0),
        'vol_outliers': outs.get('vol_outliers', 0),
    })
    return metrics
