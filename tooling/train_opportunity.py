"""Train an OpportunityPredictor from a tidy prices CSV.

CSV expected columns: timestamp,symbol,close,volume (timestamp parseable by pandas)

This script builds simple windowed features and trains a RandomForest-based predictor
when scikit-learn is available; otherwise it writes a heuristic JSON placeholder.

Example:
  python tooling/train_opportunity.py --prices-csv data/prices.csv --window 20 --output models/opportunity.pkl
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import pandas as pd

from src.features.opportunity import OpportunityPredictor, build_basic_features


def build_dataset(df: pd.DataFrame, window: int = 20):
    # expects df with columns ['timestamp','symbol','close','volume']
    df = df.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values(['symbol','timestamp'])

    X_rows = []
    y_rows = []
    # group by symbol
    for sym, g in df.groupby('symbol'):
        g = g.sort_values('timestamp')
        closes = g['close'].astype(float).reset_index(drop=True)
        vols = g['volume'] if 'volume' in g.columns else pd.Series([0.0]*len(g))
        n = len(closes)
        # for each window ending at i, produce features and label = next return
        for i in range(window, n - 1):
            window_df = pd.DataFrame({'close': closes.iloc[i - window:i+1].values, 'volume': vols.iloc[i - window:i+1].values})
            X, syms = build_basic_features({sym: window_df}, window=window)
            # X is shape (1, features)
            label = float(closes.iloc[i+1] / closes.iloc[i] - 1.0)
            X_rows.append(X[0].tolist())
            y_rows.append(label)

    if not X_rows:
        raise RuntimeError('No training samples generated; check your CSV and window size')
    Xa = np.asarray(X_rows, dtype=float)
    ya = np.asarray(y_rows, dtype=float)
    return Xa, ya


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--prices-csv', required=True)
    p.add_argument('--window', type=int, default=20)
    p.add_argument('--output', required=True)
    args = p.parse_args(argv)

    from src.utils.io import load_prices_csv
    df = load_prices_csv(args.prices_csv, dedupe='first')
    X, y = build_dataset(df, window=args.window)
    predictor = OpportunityPredictor()
    try:
        predictor.fit(X, y)
        predictor.save(args.output)
        print(f'saved model to {args.output}')
    except Exception as exc:
        # fallback: write heuristic JSON
        with open(args.output, 'w', encoding='utf-8') as fh:
            json.dump({'heuristic': True}, fh)
        print(f'scikit-learn not available or training failed ({exc}); wrote heuristic to {args.output}')


if __name__ == '__main__':
    main()
