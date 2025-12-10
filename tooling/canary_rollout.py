#!/usr/bin/env python3
"""Simple canary rollout helper.

This script performs a simulated traffic ramp for a promoted model and monitors
shadow metrics at each step. If metrics degrade beyond configured thresholds,
it performs a rollback action (logged). In a real deployment this would call
service APIs (load balancer, API gateway) to shift traffic and perform an
actual rollback (swap model pointers, etc.).

This implementation is conservative: it re-invokes `tooling/canary_promote.py`
(which downloads the model from S3 if configured) to re-evaluate metrics on
recent buffered data and decides whether to continue ramp or rollback.

Environment / args expected:
- S3_BUCKET, S3_KEY_PROD (if model stored in S3)
- Thresholds: --min-pnl, --min-sharpe, --max-dd

Usage examples:
  python tooling/canary_rollout.py --wait 10

"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from src.utils.time import now_iso
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAMP_STEPS = [1, 5, 10, 25, 50, 100]


def run_eval_and_read_metrics(tmp_out='canary_eval.json'):
    # run canary_promote to re-evaluate the current promoted model
    cmd = [sys.executable, str(ROOT / 'tooling' / 'canary_promote.py'), '--out', tmp_out]
    env = os.environ.copy()
    # ensure PYTHONPATH includes repo
    env['PYTHONPATH'] = env.get('PYTHONPATH', '.')
    try:
        subprocess.check_call(cmd, env=env)
    except subprocess.CalledProcessError:
        # canary_promote returns non-zero in some conditions; still try to read output
        pass
    if not Path(tmp_out).exists():
        return None
    with open(tmp_out, 'r') as f:
        return json.load(f)


def check_thresholds(metrics: dict, min_pnl: float, min_sharpe: float, max_dd: float) -> bool:
    # return True if metrics are acceptable
    if metrics is None:
        return False
    pnl = metrics.get('total_pnl') or metrics.get('pnl')
    sharpe = metrics.get('sharpe')
    max_drawdown = metrics.get('max_drawdown')
    if pnl is None:
        return False
    if pnl < min_pnl:
        return False
    if sharpe is None or sharpe < min_sharpe:
        return False
    if max_drawdown is not None and abs(max_drawdown) > abs(max_dd):
        return False
    return True


def perform_rollback(note: str = ''):
    # In production this should swap model pointers or call infra APIs.
    t = now_iso()
    print(f"Rollback triggered at {t}: {note}")
    # emit placeholder for infra action
    print("ACTION: rollback model routing (user must implement actual call)")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--wait', type=int, default=30, help='Seconds to wait between ramp steps')
    p.add_argument('--min-pnl', type=float, default=0.0)
    p.add_argument('--min-sharpe', type=float, default=0.5)
    p.add_argument('--max-dd', type=float, default=0.25)
    p.add_argument('--steps', nargs='*', type=int, default=None)
    args = p.parse_args(argv)

    steps = args.steps if args.steps else RAMP_STEPS
    print('Starting canary rollout simulation with steps:', steps)

    # run an initial check
    metrics = run_eval_and_read_metrics()
    print('Initial metrics:', metrics)
    ok = check_thresholds(metrics, args.min_pnl, args.min_sharpe, args.max_dd)
    if not ok:
        print('Initial metrics do not meet thresholds; aborting rollout')
        perform_rollback('initial-failure')
        return 2

    for pct in steps:
        print(f'Ramping to {pct}% traffic')
        # In real system: call API to set traffic weight to pct
        time.sleep(args.wait)
        metrics = run_eval_and_read_metrics()
        print(f'Metrics at {pct}%:', metrics)
        if not check_thresholds(metrics, args.min_pnl, args.min_sharpe, args.max_dd):
            perform_rollback(f'metric-failure-at-{pct}%')
            return 3
        print(f'{pct}% step OK; continuing')

    print('Canary rollout completed successfully — model serving at full traffic')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
