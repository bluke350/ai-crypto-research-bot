#!/usr/bin/env python3
"""Simple canary promotion helper (skeleton).

This script is a placeholder to perform shadow evaluation of a candidate model
before switching production to it. It downloads `models/opportunity-latest.pkl`
from S3 (if configured) or uses local file, runs a short shadow evaluation against
recent buffered data, and reports metrics. Integrate with your deployment system
to automate canary promotion (e.g., update service config if metric thresholds met).
"""
from __future__ import annotations

import os
from pathlib import Path
import argparse
import pickle
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'opportunity-latest.pkl'


def fetch_from_s3_if_needed():
    s3_bucket = os.environ.get('S3_BUCKET')
    s3_key = os.environ.get('S3_KEY', 'models/opportunity-latest.pkl')
    # Only attempt S3 download if bucket and credentials are available
    if s3_bucket and os.environ.get('AWS_ACCESS_KEY_ID') and not MODEL_PATH.exists():
        try:
            import boto3
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            boto3.client('s3').download_file(s3_bucket, s3_key, str(MODEL_PATH))
            print('Downloaded latest model from S3')
        except Exception as e:
            print('Failed to download from S3:', e)


def load_model(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def shadow_eval(model, df: pd.DataFrame):
    # Basic shadow eval: compute log-loss if predict_proba available
    # Use numeric feature columns only, exclude label/return
    X = df.select_dtypes(include=[float, int]).drop(columns=['label', 'return'], errors='ignore')
    y = df.get('label')
    metrics = {}

    # probabilities or predictions
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(X)[:, 1]
        p = np.clip(p, 1e-6, 1 - 1e-6)
        try:
            loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            metrics['logloss'] = float(loss)
        except Exception:
            metrics['logloss'] = None
        preds = (p > 0.5).astype(int)
    else:
        preds = model.predict(X)
        metrics['1-acc'] = 1.0 - float((preds == y).mean())

    # Backtest-based metrics when returns available
    if 'return' in df.columns:
        # returns array
        ret = df['return'].fillna(0).to_numpy()
        # positions: +1 for predict==1, -1 for predict==0
        pos = np.where(np.asarray(preds) == 1, 1.0, -1.0)
        strat = pos * ret
        cum = np.cumsum(strat)
        total_pnl = float(cum[-1])
        mean = float(np.nanmean(strat))
        std = float(np.nanstd(strat))
        sharpe = None
        if std and std > 0:
            sharpe = float(mean / std * np.sqrt(252))
        # max drawdown
        peak = np.maximum.accumulate(cum)
        drawdowns = (cum - peak) / (peak + 1e-12)
        max_dd = float(drawdowns.min())
        win_rate = float((strat > 0).mean())

        metrics.update({
            'total_pnl': total_pnl,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
        })

    return metrics


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--buffer-root', default=ROOT / 'data' / 'online_buffer')
    p.add_argument('--model-path', default=str(MODEL_PATH))
    args = p.parse_args(argv)

    fetch_from_s3_if_needed()
    mp = Path(args.model_path)
    if not mp.exists():
        # fallback to local model if latest not present
        alt = ROOT / 'models' / 'opportunity.pkl'
        if alt.exists():
            mp = alt
        else:
            print('Model not found at', mp)
            return 2

    model = load_model(mp)
    # sample recent data
    from src.data.online_buffer import sample_recent
    df = sample_recent(max_rows=1000)
    if df.empty:
        print('No recent data for shadow eval')
        return 3

    metrics = shadow_eval(model, df.tail(200))
    print('Shadow eval metrics:', json.dumps(metrics))
    # In a real system, compare to thresholds and trigger deployment
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
