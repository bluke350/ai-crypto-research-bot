from __future__ import annotations

import typing as _t
import math

import pandas as pd
import numpy as np


def volatility_rank(prices: pd.DataFrame, window: int = 60) -> float:
    """Return realized volatility (std of returns) over the window expressed as annualized (sqrt(252)).

    Returns 0.0 for empty or insufficient series.
    """
    if prices is None or prices.empty or "close" not in prices.columns:
        return 0.0
    close = prices["close"].astype(float)
    if len(close) < 2:
        return 0.0
    # compute returns and take last `window` returns
    returns = close.pct_change().dropna().iloc[-window:]
    if returns.empty:
        return 0.0
    rv = float(returns.std())
    # annualize assuming 252 trading days and per-bar interval approximate by 1/day if bars > 390 etc.
    return rv * math.sqrt(252)


def liquidity_filter(prices: pd.DataFrame, min_volume: float = 0.0) -> bool:
    """Return True if median volume >= min_volume. Handles missing volume column by returning False when min_volume > 0.
    """
    if prices is None or prices.empty or "volume" not in prices.columns:
        return min_volume <= 0.0
    med = float(prices["volume"].astype(float).median())
    return med >= float(min_volume)


def correlation_rank(prices_df: pd.DataFrame) -> _t.Dict[str, float]:
    """Return average absolute correlation to other series for each column in a multi-column DataFrame of closes.

    Input: DataFrame with columns for symbols containing close prices aligned by index.
    Output: dict symbol -> mean absolute correlation (lower is better). If only one column, returns {symbol:0.0}.
    """
    if prices_df is None or prices_df.empty or not isinstance(prices_df, pd.DataFrame):
        return {}
    # compute returns for each column
    rets = prices_df.pct_change().dropna(how="all")
    cols = list(rets.columns)
    if len(cols) <= 1:
        return {cols[0]: 0.0} if cols else {}
    corr = rets.corr().abs()
    out = {}
    for c in cols:
        # exclude self-correlation by taking mean of off-diagonal values
        others = [v for k, v in corr[c].items() if k != c]
        out[c] = float(np.nanmean(others) if others else 0.0)
    return out


def rank_universe(prices_by_symbol: _t.Dict[str, pd.DataFrame], min_volume: float = 0.0, vol_window: int = 60) -> _t.List[str]:
    """Rank symbols by combined heuristic: liquidity pass, then ascending volatility and correlation.

    Returns list of symbol names sorted ascending by score where lower is better.
    Score = normalized volatility + normalized correlation - liquidity_bonus (higher liquidity reduces score).
    """
    if not prices_by_symbol:
        return []
    # prepare stats
    vol_scores = {}
    liq_ok = {}
    closes = {}
    for s, df in prices_by_symbol.items():
        vol_scores[s] = volatility_rank(df, window=vol_window)
        liq_ok[s] = liquidity_filter(df, min_volume=min_volume)
        if df is None or df.empty or "close" not in df.columns:
            closes[s] = pd.Series(dtype=float)
        else:
            closes[s] = df["close"].astype(float)

    # correlations want a DataFrame of closes aligned
    close_df = pd.DataFrame(closes).replace({np.nan: None}).dropna(how="all")
    # correlation rank zero if no other data
    corr_scores = correlation_rank(close_df) if not close_df.empty else {k: 0.0 for k in closes.keys()}

    # normalize vol and corr to 0..1 range
    vol_vals = np.array(list(vol_scores.values()), dtype=float)
    corr_vals = np.array([corr_scores.get(k, 0.0) for k in vol_scores.keys()], dtype=float)
    if len(vol_vals) == 0:
        return []
    vol_min, vol_max = float(vol_vals.min()), float(vol_vals.max())
    corr_min, corr_max = float(corr_vals.min()), float(corr_vals.max())
    # avoid div by zero
    def norm(val, lo, hi):
        if hi - lo <= 1e-12:
            return 0.0
        return (val - lo) / (hi - lo)

    scores = {}
    for i, s in enumerate(vol_scores.keys()):
        v = vol_scores[s]
        c = corr_scores.get(s, 0.0)
        nv = norm(v, vol_min, vol_max)
        nc = norm(c, corr_min, corr_max)
        liquidity_bonus = 0.2 if liq_ok.get(s, False) else 0.0
        scores[s] = nv + nc - liquidity_bonus

    # return symbols sorted ascending by score
    return sorted(scores.keys(), key=lambda k: scores[k])
