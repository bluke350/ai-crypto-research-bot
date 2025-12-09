from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def btc_dominance(market_caps: pd.Series) -> float:
    """Compute BTC dominance given a Series of market caps indexed by symbol.

    Expects `market_caps` where index contains 'BTC' and other assets and values are market cap numbers.
    Returns BTC dominance in [0,1].
    """
    if 'BTC' not in market_caps.index:
        raise ValueError('market_caps must contain BTC')
    total = float(market_caps.sum())
    if total <= 0:
        return 0.0
    return float(market_caps.loc['BTC'] / total)


def rolling_correlation_with_btc(prices: pd.Series, btc_prices: pd.Series, window: int = 30) -> pd.Series:
    """Return rolling Pearson correlation between `prices` and `btc_prices` with a given window."""
    if len(prices) != len(btc_prices):
        # align by index
        df = pd.concat([prices, btc_prices], axis=1, join='inner')
        df.columns = ['p', 'btc']
    else:
        df = pd.concat([prices, btc_prices], axis=1)
        df.columns = ['p', 'btc']
    # use fill_method=None to avoid pandas' deprecated default behavior
    return df['p'].pct_change(fill_method=None).rolling(window).corr(df['btc'].pct_change(fill_method=None)).fillna(0.0)


def lagged_cross_correlation(a: pd.Series, b: pd.Series, max_lag: int = 5) -> Tuple[int, float]:
    """Compute lagged cross-correlation between series `a` and `b`.

    Returns (best_lag, corr) where best_lag is in [-max_lag..max_lag] indicating how much `b` leads a (positive means b leads a).
    """
    # align
    df = pd.concat([a, b], axis=1, join='inner')
    df.columns = ['a', 'b']
    best = (0, 0.0)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            shifted = df['b'].shift(-lag)  # b leads a by -lag
            common = df['a'].copy()
        else:
            shifted = df['b'].shift(-lag)
            common = df['a']
        corr = common.pct_change(fill_method=None).corr(shifted.pct_change(fill_method=None))
        if pd.isna(corr):
            corr = 0.0
        if abs(corr) > abs(best[1]):
            best = (lag, float(corr))
    return best


def cross_pair_features(prices_a: pd.Series, prices_b: pd.Series, window: int = 30, max_lag: int = 5) -> Dict[str, float]:
    """Compute a small set of cross-pair features between two price series.

    Returns a dict with rolling correlation mean/std and lagged correlation best lag and value.
    """
    # rolling correlations
    roll_corr = rolling_correlation_with_btc(prices_a, prices_b, window=window) if not prices_b.empty else pd.Series([0.0])
    mean_corr = float(roll_corr.mean()) if not roll_corr.empty else 0.0
    std_corr = float(roll_corr.std()) if not roll_corr.empty else 0.0
    best_lag, best_corr = lagged_cross_correlation(prices_a, prices_b, max_lag=max_lag)
    return {
        'rolling_corr_mean': mean_corr,
        'rolling_corr_std': std_corr,
        'lagged_best_lag': int(best_lag),
        'lagged_best_corr': float(best_corr),
    }


__all__ = ['btc_dominance', 'rolling_correlation_with_btc', 'lagged_cross_correlation', 'cross_pair_features']
