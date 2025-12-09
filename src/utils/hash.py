from __future__ import annotations
import hashlib
import pandas as pd


def dataframe_hash(df: pd.DataFrame) -> str:
    """Return a stable SHA256 hex digest for a DataFrame's values and columns.

    This uses CSV serialization with index=False to create a reproducible
    byte representation for hashing. Not cryptographically secure for large
    data but fine for run artifact reproducibility.
    """
    if df is None:
        return ""
    try:
        # ensure deterministic column order
        cols = list(df.columns)
        buf = df.to_csv(index=False, columns=cols).encode("utf-8")
        return hashlib.sha256(buf).hexdigest()
    except Exception:
        # fallback: hash repr
        return hashlib.sha256(repr(df).encode("utf-8")).hexdigest()
