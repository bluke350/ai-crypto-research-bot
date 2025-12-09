from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple


REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        raise ValueError("DataFrame is required")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.empty:
        raise ValueError("DataFrame is empty")
    df = df.copy()
    # ensure timestamp is datetime and UTC
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute RSI and return DataFrame with column `rsi_{window}`.

    Parameters
    ----------
    df : DataFrame
        Input OHLCV dataframe with required columns.
    window : int
        Lookback window for RSI.

    Returns
    -------
    DataFrame
        DataFrame with the RSI column aligned to input timestamps.

    Raises
    ------
    ValueError
        If input is invalid.
    """
    df = _validate(df)
    close = df["close"].astype(float)
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / window, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi_series = 100 - (100 / (1 + rs))
    col = f"rsi_{window}"
    out = pd.DataFrame({"timestamp": df["timestamp"], col: rsi_series})
    return out


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Compute MACD (fast - slow) and signal line.

    Returns columns: `macd_{fast}_{slow}`, `macd_signal_{signal}`
    """
    df = _validate(df)
    close = df["close"].astype(float)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({
        "timestamp": df["timestamp"],
        f"macd_{fast}_{slow}": macd_line,
        f"macd_signal_{signal}": sig,
    })


def atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute Average True Range (ATR).

    Returns column: `atr_{window}`
    """
    df = _validate(df)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1.0 / window, adjust=False).mean()
    return pd.DataFrame({"timestamp": df["timestamp"], f"atr_{window}": atr_series})


def bollinger(df: pd.DataFrame, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Compute Bollinger Bands: middle, upper, lower.

    Columns: `bb_mid_{window}`, `bb_upper_{window}_{k}`, `bb_lower_{window}_{k}`
    """
    df = _validate(df)
    close = df["close"].astype(float)
    mid = close.rolling(window=window, min_periods=1).mean()
    std = close.rolling(window=window, min_periods=1).std()
    upper = mid + k * std
    lower = mid - k * std
    return pd.DataFrame({
        "timestamp": df["timestamp"],
        f"bb_mid_{window}": mid,
        f"bb_upper_{window}_{k}": upper,
        f"bb_lower_{window}_{k}": lower,
    })


def realized_volatility(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Compute realized volatility as rolling std of log returns.

    Returns column: `rv_{window}` (per-period volatility, not annualized).
    """
    df = _validate(df)
    close = df["close"].astype(float)
    logret = np.log(close).diff()
    rv = logret.rolling(window=window, min_periods=1).std()
    return pd.DataFrame({"timestamp": df["timestamp"], f"rv_{window}": rv})


def rolling_volume_stats(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Compute rolling volume mean and z-score as lightweight microstructure features.

    Returns columns: `vol_mean_{window}`, `vol_z_{window}`
    """
    df = _validate(df)
    vol = df["volume"].astype(float)
    vm = vol.rolling(window=window, min_periods=1).mean()
    vs = vol.rolling(window=window, min_periods=1).std().replace(0, np.nan)
    z = (vol - vm) / vs
    return pd.DataFrame({
        "timestamp": df["timestamp"],
        f"vol_mean_{window}": vm,
        f"vol_z_{window}": z.fillna(0.0),
    })


def all_technical(df: pd.DataFrame, rsi_window=14, macd_params=(12, 26, 9), atr_window=14, bollinger_params=(20, 2.0)) -> pd.DataFrame:
    """Return a DataFrame joining common technical indicators aligned by timestamp."""
    parts = [
        rsi(df, rsi_window),
        macd(df, *macd_params),
        atr(df, atr_window),
        bollinger(df, *bollinger_params),
        realized_volatility(df, window=30),
        rolling_volume_stats(df, window=30),
    ]
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="timestamp", how="outer")
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out
