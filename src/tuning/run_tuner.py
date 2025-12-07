"""CLI wrapper to run the small Optuna tuner on a demo objective.

This script is intentionally minimal and usable as an example for integrating
the tuner into walk-forward or simulator-based evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tuning.optimizers import run_optuna_tuner


def demo_objective(params: dict) -> float:
    # toy objective: prefers action_scale near 3.0 and ent_coef near 0.01
    scale = float(params.get('action_scale', 1.0))
    ent = float(params.get('ent_coef', 0.01))
    # negative of squared distance (we maximize)
    return -((scale - 3.0)**2) - ((ent - 0.01)**2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    search_space = {
        'action_scale': ('uniform', 0.5, 6.0),
        'ent_coef': ('loguniform', 1e-4, 1e-1),
    }

    study = run_optuna_tuner(demo_objective, search_space, n_trials=args.trials, seed=args.seed)
    best = study.best_trial
    out = {'value': best.value, 'params': best.params}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding='utf-8')
        print(f"wrote {args.out}")
    else:
        print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
