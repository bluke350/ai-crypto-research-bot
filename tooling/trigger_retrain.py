"""CLI to test the OnlineRetrainer scaffold.

Example usages:
  # trigger retrain if metrics indicate degradation
  python tooling/trigger_retrain.py --metrics-file examples/metrics.json --metric val_loss --threshold 0.05

The script will use `example_retrain_fn` by default to create a dummy artifact and log it
via `RunLogger`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from src.training.online_retrainer import OnlineRetrainer, RetrainPolicy, example_retrain_fn


def read_metrics_file(p: Path):
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metrics-file', type=str, help='Path to a JSON file with latest metrics (a dict).')
    p.add_argument('--metric', type=str, default='val_loss', help='Metric name to monitor')
    p.add_argument('--direction', type=str, choices=['down', 'up'], default='down')
    p.add_argument('--threshold', type=float, default=0.02)
    p.add_argument('--window', type=int, default=5)
    p.add_argument('--cooldown', type=int, default=3600)
    args = p.parse_args()

    metrics = read_metrics_file(Path(args.metrics_file)) if args.metrics_file else {}
    policy = RetrainPolicy(metric=args.metric, direction=args.direction, threshold=args.threshold, window=args.window, cooldown_seconds=args.cooldown)
    retrainer = OnlineRetrainer(retrain_fn=example_retrain_fn, policy=policy)

    triggered = retrainer.ingest(metrics)
    if triggered:
        print('Retrain triggered and artifact logged')
    else:
        print('No retrain triggered')


if __name__ == '__main__':
    main()
