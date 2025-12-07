from __future__ import annotations

from typing import Dict, Tuple
import pandas as pd
import numpy as np


def pairwise_lagged_correlation_matrix(series_dict: Dict[str, pd.Series], max_lag: int = 5) -> Dict[Tuple[str, str], Tuple[int, float]]:
    """Compute pairwise best lagged correlation between all pairs.

    Returns a dict keyed by (a,b) -> (best_lag, corr) where best_lag means b leads a by that lag (positive -> b leads a).
    """
    keys = list(series_dict.keys())
    out = {}
    for i, ka in enumerate(keys):
        for j, kb in enumerate(keys):
            if ka == kb:
                out[(ka, kb)] = (0, 1.0)
                continue
            a = series_dict[ka].dropna()
            b = series_dict[kb].dropna()
            # align on overlapping index
            df = pd.concat([a, b], axis=1, join='inner')
            if df.shape[0] < 3:
                out[(ka, kb)] = (0, 0.0)
                continue
            df.columns = ['a', 'b']
            # use returns to avoid level-driven correlation
            ra = df['a'].pct_change().dropna()
            rb = df['b'].pct_change()
            best = (0, 0.0)
            for lag in range(-max_lag, max_lag + 1):
                # shifting rb: positive lag means b leads a by `lag` steps
                shifted_rb = rb.shift(-lag)
                # align with ra
                common = pd.concat([ra, shifted_rb], axis=1, join='inner').dropna()
                if common.shape[0] < 3:
                    continue
                corr = common.iloc[:, 0].corr(common.iloc[:, 1])
                if pd.isna(corr):
                    corr = 0.0
                if abs(corr) > abs(best[1]):
                    best = (lag, float(corr))
            out[(ka, kb)] = best
    return out


def build_lagged_features(target: pd.Series, others: Dict[str, pd.Series], max_lag: int = 5) -> pd.DataFrame:
    """Create lagged features for `target` from other series.

    For each other series, include lagged returns for lags in 1..max_lag where the other leads target.
    Returns a DataFrame indexed like `target` with columns like `{pair}_lag{lag}`.
    """
    # align all series to the target index
    df_all = pd.DataFrame(index=target.index)
    df_all['target'] = target
    for name, s in others.items():
        # reindex other series to the target index to keep alignment
        df_all[name] = s.reindex(target.index)
    # drop rows where target is NaN
    df_all = df_all.dropna(subset=['target'])
    if df_all.empty:
        return pd.DataFrame(index=target.index)
    features = pd.DataFrame(index=df_all.index)
    for name in others.keys():
        # compute lagged cross-correlation to find useful lead/lag
        best_lag = 0
        best_corr = 0.0
        for lag in range(1, max_lag + 1):
            shifted = df_all[name].shift(-lag)
            ta = df_all['target'].pct_change().dropna()
            tb = shifted.pct_change().dropna()
            # align
            common = pd.concat([ta, tb], axis=1, join='inner').dropna()
            if common.shape[0] < 3:
                continue
            corr = common.iloc[:, 0].corr(common.iloc[:, 1])
            if pd.isna(corr):
                continue
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        # if best_lag > 0, create lagged features (b leads target by best_lag)
        if best_lag > 0:
            col = f"{name}_lag{best_lag}"
            feat = df_all[name].shift(-best_lag).pct_change()
            features[col] = feat.reindex(features.index).fillna(0.0)
        else:
            # fallback: include contemporaneous returns
            col = f"{name}_ret"
            features[col] = df_all[name].pct_change().reindex(features.index).fillna(0.0)
    return features


__all__ = ['pairwise_lagged_correlation_matrix', 'build_lagged_features']
