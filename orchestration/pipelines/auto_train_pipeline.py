from __future__ import annotations

import argparse
import json
import os
import uuid
import logging
from typing import Optional, Dict, Any, List

import pandas as pd

from src.utils.config import load_yaml
from src.validation.walk_forward import evaluate_walk_forward
from src.tuning.optimizers import BayesianTuner, RandomSearchTuner
from src.execution.simulator import Simulator
from src.models.rl.policy_adapter import RLPolicyStrategy
from src.ingestion.providers import kraken_rest
from src.persistence.db import RunLogger

LOG = logging.getLogger(__name__)


def load_prices(prices_csv: Optional[str], symbol: Optional[str]) -> pd.DataFrame:
    if prices_csv:
        from src.utils.io import load_prices_csv
        # default dedupe behavior is 'first' for backward compatibility
        return load_prices_csv(prices_csv, dedupe='first')
    try:
        sym = symbol or (load_yaml('configs/kraken.yaml').get('kraken', {}).get('symbols', ['XBT/USD'])[0])
        return kraken_rest.get_ohlc(sym, '1m', since=0)
    except Exception as e:
        LOG.warning('failed to load prices: %s', e)
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', type=str, default=None)
    p.add_argument('--prices-csv', type=str, default=None)
    p.add_argument('--dedupe', type=str, default='first', choices=('none','first','last','mean'), help='How to handle duplicate timestamps in CSV: none|first|last|mean')
    p.add_argument('--optimizer', type=str, default='bayes', choices=('bayes', 'random', 'optuna'))
    p.add_argument('--optuna-db', type=str, default=None, help='Optuna DB URL (sqlite:///path) to use for study storage')
    p.add_argument('--optuna-pruner', type=str, default='median', choices=('median', 'asha'), help='Optuna pruner to use when optimizer=optuna')
    p.add_argument('--param-space', type=str, default=None)
    p.add_argument('--n-trials', type=int, default=10)
    p.add_argument('--seeds', type=str, default='0')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--window', type=int, default=100)
    p.add_argument('--step', type=int, default=25)
    p.add_argument('--output', type=str, default='experiments/artifacts')
    p.add_argument('--per-regime', action='store_true')
    p.add_argument('--register', action='store_true')
    args = p.parse_args()

    prices = load_prices(args.prices_csv, args.symbol)
    # if user explicitly requested a dedupe mode, and we loaded from CSV, apply it
    if getattr(args, 'dedupe', None) and args.prices_csv and args.dedupe != 'first':
        try:
            # re-run loader with requested dedupe
            from src.utils.io import load_prices_csv
            prices = load_prices_csv(args.prices_csv, dedupe=args.dedupe)
        except Exception:
            LOG.exception('failed to apply dedupe mode %s', args.dedupe)
    # if user explicitly requested a dedupe mode, and we loaded from CSV, apply it
    if getattr(args, 'dedupe', None) and args.prices_csv and args.dedupe != 'first':
        try:
            if args.dedupe == 'none':
                pass
            elif args.dedupe in ('first','last'):
                prices = prices.drop_duplicates(subset=['timestamp'], keep=args.dedupe).reset_index(drop=True)
            elif args.dedupe == 'mean':
                num_cols = prices.select_dtypes(include=['number']).columns.tolist()
                other_cols = [c for c in prices.columns if c not in (num_cols + ['timestamp'])]
                agg_dict = {c: 'mean' for c in num_cols}
                for c in other_cols:
                    agg_dict[c] = 'last'
                prices = prices.groupby('timestamp', sort=True, as_index=False).agg(agg_dict)
        except Exception:
            LOG.exception('failed to apply dedupe mode %s', args.dedupe)

    if args.param_space:
        param_space = json.loads(args.param_space)
    else:
        param_space = {'short': [5, 10], 'long': [20, 40], 'size': [1.0]}

    # choose tuner
    if args.optimizer == 'random':
        tuner = RandomSearchTuner(param_space, n_trials=args.n_trials)
    elif args.optimizer == 'optuna':
        try:
            from src.tuning.optimizers import OptunaTuner
        except Exception:
            raise RuntimeError("optuna not available; install optuna or choose a different optimizer")
        tuner = OptunaTuner(param_space, n_trials=args.n_trials, db_url=args.optuna_db, pruner=args.optuna_pruner)
    else:
        tuner = BayesianTuner(param_space, n_trials=args.n_trials)

    sim = lambda: Simulator(seed=0)
    # run tuning to get best params
    try:
        # prepare a simple strategy_factory wrapper for MovingAverageCrossover to be used by the tuner objective
        def ma_factory(**p):
            return __import__('src.strategies.moving_average', fromlist=['MovingAverageCrossover']).MovingAverageCrossover(**p)

        def _objective(params):
            out = evaluate_walk_forward(prices, targets=None, simulator=sim, window=args.window, step=args.step, tuner=None, param_space=None, strategy_factory=ma_factory)
            # objective: return average final_value across folds (higher better)
            folds = out.get('folds', [])
            if not folds:
                return float('-inf')
            vals = [f['metrics'].get('final_value', 0.0) for f in folds]
            return float(sum(vals) / len(vals))

        best_params, best_score = tuner.optimize(_objective)
    except Exception:
        # fallback: no tuning
        best_params, best_score = ({}, None)

    # train RL for seeds
    seeds = [int(s) for s in str(args.seeds).split(',') if str(s).strip()]
    os.makedirs(args.output, exist_ok=True)
    run_id = str(uuid.uuid4())
    out_dir = os.path.join(args.output, run_id)
    os.makedirs(out_dir, exist_ok=True)

    model_paths: List[str] = []
    for s in seeds:
        # construct a unique save path for seed
        save_path = os.path.join(out_dir, f'ppo_seed{s}.pth')
        # call train_ppo CLI programmatically
        try:
            from src.models.rl.train_ppo import main as train_ppo_main
            import sys as _sys
            argv = [ 'train_ppo.py', '--steps', str(args.steps), '--save', save_path, '--seed', str(s) ]
            prev = list(_sys.argv)
            try:
                _sys.argv = argv
                train_ppo_main()
            finally:
                _sys.argv = prev
            model_paths.append(save_path)
        except Exception as e:
            LOG.exception('failed training for seed %s: %s', s, e)

    # evaluate trained models using walk-forward and RLPolicyStrategy
    eval_results = {}
    for mp in model_paths:
        try:
            strat_fact = lambda **p: RLPolicyStrategy(ckpt_path=mp, **p)
            out = evaluate_walk_forward(prices, targets=None, simulator=sim, window=args.window, step=args.step, tuner=None, param_space=None, strategy_factory=strat_fact)
            eval_results[mp] = out
        except Exception:
            LOG.exception('failed to run evaluation for model %s', mp)

    # register to DB
    if args.register:
        with RunLogger(run_id, cfg={"param_space": param_space, "best_params": best_params, "seeds": seeds}) as rl:
            for p in model_paths:
                try:
                    rl.log_artifact(p, kind='checkpoint')
                except Exception:
                    LOG.exception('failed to log artifact %s', p)
            # log summary metrics per model
            for mp, out in eval_results.items():
                try:
                    # compute basic metric: mean final_value across folds
                    vals = [f['metrics']['final_value'] for f in out.get('folds', [])]
                    mean_val = float(sum(vals)/len(vals)) if vals else 0.0
                    rl.log_metrics({f"mean_final_value_{os.path.basename(mp)}": mean_val})
                except Exception:
                    LOG.exception('failed to log metrics for %s', mp)

    # write artifacts
    with open(os.path.join(out_dir, 'eval.json'), 'w', encoding='utf-8') as fh:
        json.dump({'best_params': best_params, 'best_score': best_score, 'models': model_paths, 'eval': eval_results}, fh, indent=2)

    print(f'Auto-train completed; artifacts in {out_dir}')


if __name__ == '__main__':
    main()
