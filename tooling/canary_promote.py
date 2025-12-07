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
    if s3_bucket and not MODEL_PATH.exists():
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
    X = df.drop(columns=['label'], errors='ignore')
    y = df.get('label')
    metrics = {}
    if 'return' in df.columns:
        # simple simulated PnL: take sign(pred - 0.5) * return and report mean
        if hasattr(model, 'predict_proba'):
            p = model.predict_proba(X)[:, 1]
        else:
            p = model.predict(X)
        p = np.asarray(p).ravel()
        preds_sign = np.sign(p - 0.5)
        pnl = (preds_sign * df['return'].values).mean()
        metrics['pnl'] = float(pnl)

    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(X)[:, 1]
        p = np.clip(p, 1e-6, 1 - 1e-6)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        metrics['logloss'] = float(loss)
    else:
        preds = model.predict(X)
        acc = float((preds == y).mean())
        metrics['1-acc'] = 1.0 - acc

    return metrics


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--buffer-root', default=ROOT / 'data' / 'online_buffer')
    p.add_argument('--model-path', default=str(MODEL_PATH))
    args = p.parse_args(argv)

    fetch_from_s3_if_needed()
    mp = Path(args.model_path)
    if not mp.exists():
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
