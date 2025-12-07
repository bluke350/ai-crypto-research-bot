from __future__ import annotations

from typing import Literal
import logging
import pandas as pd

LOG = logging.getLogger(__name__)


def load_prices_csv(path: str, dedupe: Literal["none", "first", "last", "mean"] = "first") -> pd.DataFrame:
    """Load a CSV of OHLCV prices and optionally dedupe by `timestamp`.

    Parameters
    ----------
    path: str
        Path to CSV file.
    dedupe:
        How to handle duplicate timestamps: 'none', 'first', 'last', 'mean'.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with a parsed `timestamp` column when present.
    """
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        dup_mask = df["timestamp"].duplicated(keep=False)
        if dup_mask.any():
            LOG.warning("Duplicate timestamps detected in %s; applying dedupe=%s", path, dedupe)
            if dedupe == "none":
                # leave as-is
                pass
            elif dedupe in ("first", "last"):
                df = df.drop_duplicates(subset=["timestamp"], keep=dedupe).reset_index(drop=True)
            elif dedupe == "mean":
                num_cols = df.select_dtypes(include=["number"]).columns.tolist()
                other_cols = [c for c in df.columns if c not in (num_cols + ["timestamp"])]
                agg_dict = {c: "mean" for c in num_cols}
                for c in other_cols:
                    agg_dict[c] = "last"
                df = df.groupby("timestamp", sort=True, as_index=False).agg(agg_dict)
            else:
                LOG.warning("Unknown dedupe mode '%s' - falling back to 'first'", dedupe)
                df = df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    return df
