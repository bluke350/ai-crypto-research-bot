#!/usr/bin/env python3
"""Incremental online updater for opportunity model (MVP).

Usage (example):
  python tooling/online_update.py --buffer-root data/online_buffer --model models/opportunity.pkl --out models/opportunity-updated.pkl --min-rows 200

This script is intentionally conservative: it only updates models that provide a `partial_fit` method
or scikit-learn estimators that support `partial_fit`. For other models, it will exit.

It uses `river`'s ADWIN change detector on validation loss to gate promotions.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle
import tempfile
import json

import numpy as np
import pandas as pd

try:
    from river.drift import ADWIN
except Exception:
    # Graceful fallback if river isn't installed in the environment.
    class ADWIN:
        def __init__(self):
            self.change_detected = False
        def update(self, _):
            return

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUFFER_ROOT = ROOT / "data" / "online_buffer"


def load_model(path: Path):
    # Support: pickle scikit-learn models (.pkl) and PyTorch state_dict (.pth)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix in ('.pkl', '.pickle'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    if path.suffix in ('.pth', '.pt'):
        # lazy import torch
        try:
            import torch
            import json
        except Exception:
            raise RuntimeError('PyTorch required to load .pth models')
        # Expect a metadata file alongside the pth
        meta_path = path.with_suffix('.meta.json')
        if not meta_path.exists():
            raise RuntimeError('Missing metadata for PyTorch model: %s' % meta_path)
        meta = json.loads(meta_path.read_text())
        arch = meta.get('arch')
        if arch == 'SimpleNet':
            # reconstruct the same architecture
            import torch.nn as nn

            class SimpleNet(nn.Module):
                def __init__(self, input_dim: int = 3, hidden: int = 16):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, 1),
                        nn.Sigmoid(),
                    )

                def forward(self, x):
                    return self.net(x)

            model = SimpleNet(input_dim=meta.get('input_dim', 3), hidden=meta.get('hidden', 16))
            state = torch.load(path)
            model.load_state_dict(state)
            model.eval()
            return model
        raise RuntimeError('Unknown PyTorch arch: %s' % arch)
    # fallback: attempt to pickle-load
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_model(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def prepare_X_y(df: pd.DataFrame, label_col: str = 'label'):
    if df.empty:
        return None, None
    X = df.drop(columns=[label_col], errors='ignore')
    if label_col in df.columns:
        y = df[label_col]
    else:
        y = None
    return X, y


def evaluate_predict(model, X: pd.DataFrame, y: pd.Series):
    # simple binary log-loss as metric if probabilities available, otherwise misclassification
    if y is None or X is None or X.empty:
        return None
    try:
        if hasattr(model, 'predict_proba'):
            p = model.predict_proba(X)[:, 1]
            # clip
            p = np.clip(p, 1e-6, 1 - 1e-6)
            loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            return float(loss)
        else:
            preds = model.predict(X)
            acc = (preds == y).mean()
            return float(1 - acc)
    except Exception:
        return None


def try_partial_fit(model, X, y, classes=None, epochs=1):
    # Try to use partial_fit if available
    if not hasattr(model, 'partial_fit'):
        # For PyTorch models, try a tiny fine-tune loop if torch available
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except Exception:
            return False
        # Expect a torch.nn.Module
        if isinstance(model, torch.nn.Module):
            model.train()
            X_t = torch.tensor(X.values.astype('float32'))
            if y is None:
                return False
            y_t = torch.tensor(y.values.astype('float32')).unsqueeze(1)
            opt = optim.Adam(model.parameters(), lr=1e-4)
            loss_fn = nn.BCELoss()
            for _ in range(epochs):
                opt.zero_grad()
                out = model(X_t)
                loss = loss_fn(out, y_t)
                loss.backward()
                opt.step()
            model.eval()
            return True
        return False
    for _ in range(epochs):
        try:
            if classes is not None:
                model.partial_fit(X, y, classes=classes)
            else:
                model.partial_fit(X, y)
        except TypeError:
            # some implementations expect numpy arrays
            model.partial_fit(X.values, y.values)
    return True


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--buffer-root', type=Path, default=DEFAULT_BUFFER_ROOT)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--min-rows', type=int, default=200)
    p.add_argument('--label-col', type=str, default='label')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--replay-frac', type=float, default=0.2)
    args = p.parse_args(argv)

    # load buffer data
    from src.data.online_buffer import sample_recent

    df = sample_recent(max_rows=2000)
    if df.empty or len(df) < args.min_rows:
        print('Not enough buffer rows: found', len(df))
        return 0

    # split into train/val (simple time split)
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].reset_index(drop=True)
    val_df = df.iloc[split:].reset_index(drop=True)

    X_train, y_train = prepare_X_y(train_df, label_col=args.label_col)
    X_val, y_val = prepare_X_y(val_df, label_col=args.label_col)

    # load model
    model = load_model(args.model)

    # baseline evaluation
    base_metric = evaluate_predict(model, X_val, y_val)
    print('Base metric (lower is better):', base_metric)

    # prepare classes if possible
    classes = None
    if y_train is not None:
        classes = np.unique(y_train)

    # small replay support: sample a fraction from the head of train
    replay_n = int(len(train_df) * args.replay_frac)
    if replay_n > 0 and replay_n < len(train_df):
        replay_df = train_df.sample(replay_n)
        X_replay, y_replay = prepare_X_y(replay_df, label_col=args.label_col)
        if X_replay is not None and y_replay is not None:
            # append to the end of X_train to mix
            X_train = pd.concat([X_train, X_replay], ignore_index=True)
            y_train = pd.concat([y_train, y_replay], ignore_index=True)

    # attempt partial_fit
    updated = False
    if try_partial_fit(model, X_train, y_train, classes=classes, epochs=args.epochs):
        print('Model updated with partial_fit')
        updated = True
    else:
        print('Model does not support partial_fit; skipping incremental update')

    if not updated:
        return 0

    # evaluate and apply ADWIN gating on validation loss
    new_metric = evaluate_predict(model, X_val, y_val)
    print('New metric:', new_metric)
    if new_metric is None or base_metric is None:
        print('Could not compute metrics; not promoting')
        return 0

    # ADWIN monitors whether the metric is improving; we look for a sustained decrease
    adwin = ADWIN()
    adwin.update(base_metric)
    adwin.update(new_metric)
    # If ADWIN signals change and new_metric < base_metric then accept
    promote = (new_metric < base_metric) and adwin.change_detected
    if promote:
        print('Promotion gate passed; saving model to', args.out)
        save_model(model, args.out)
        # optionally also upload to S3 if configured
        try:
            import boto3
            s3_bucket = os.environ.get('S3_BUCKET')
            s3_key = os.environ.get('S3_KEY', 'models/opportunity-online-updated.pkl')
            if s3_bucket:
                s3 = boto3.client('s3')
                s3.upload_file(str(args.out), s3_bucket, s3_key)
                print('Uploaded updated model to s3://%s/%s' % (s3_bucket, s3_key))
        except Exception:
            pass
    else:
        print('Promotion gate failed; not saving model')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
