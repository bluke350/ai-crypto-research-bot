"""Register a training run into the experiments registry.

Usage examples:
  # create a run with a config and metrics
  python tooling/register_training_run.py --config configs/project.yaml --metrics '{"val_loss":0.12,"acc":0.91}' --artifact models/ppo_latest.pth

  # auto-generate run id and read metrics from a json file
  python tooling/register_training_run.py --metrics-file examples/metrics.json --artifact models/ppo_latest.pth

This script uses `src.persistence.db.RunLogger` to persist metadata into
the default experiment DB (`sqlite:///experiments/registry.db`) unless
`--db-url` is specified.
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

from src.persistence.db import RunLogger


def read_config(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    text = Path(path).read_text(encoding='utf-8')
    try:
        return json.loads(text)
    except Exception:
        # try YAML if installed
        try:
            import yaml

            return yaml.safe_load(text)
        except Exception:
            return {"raw": text}


def read_metrics(args: argparse.Namespace) -> Dict[str, float]:
    if args.metrics:
        try:
            return json.loads(args.metrics)
        except Exception:
            return {}
    if args.metrics_file:
        return json.loads(Path(args.metrics_file).read_text(encoding='utf-8'))
    return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', type=str, help='Optional run id. If omitted, generates a UUID.')
    p.add_argument('--config', type=str, help='Path to a JSON or YAML config describing the run.')
    p.add_argument('--metrics', type=str, help='JSON string of metrics to log, e.g. "{\"acc\":0.9}"')
    p.add_argument('--metrics-file', type=str, help='Path to a JSON file with metrics.')
    p.add_argument('--artifact', type=str, action='append', help='Path to artifact file to register (can pass multiple).')
    p.add_argument('--kind', type=str, default='model', help='Artifact kind (model, report, checkpoint).')
    p.add_argument('--db-url', type=str, help='Optional SQLAlchemy DB URL to override default.')

    args = p.parse_args()

    run_id = args.run_id or str(uuid.uuid4())
    cfg = read_config(Path(args.config)) if args.config else None
    metrics = read_metrics(args)

    with RunLogger(run_id=run_id, cfg=cfg or {}, db_url=args.db_url) as rl:
        if metrics:
            rl.log_metrics(metrics)
        if args.artifact:
            for a in args.artifact:
                rl.log_artifact(a, kind=args.kind)

    print(f"Registered run {run_id} into the experiment DB (db_url={args.db_url or 'default'})")


if __name__ == '__main__':
    main()
