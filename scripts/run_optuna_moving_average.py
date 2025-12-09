#!/usr/bin/env python3
"""Run a short Optuna study to tune MovingAverageCrossover parameters using walk-forward evaluation.

Outputs a JSON file `optuna_study_summary.json` in the artifact dir with best params and best value.
"""
import json
import os
import uuid
from typing import Dict, Any

from src.utils.io import load_prices_csv
from src.validation.walk_forward import evaluate_walk_forward
from src.execution.simulator import Simulator
from src.strategies.moving_average import MovingAverageCrossover
from src.tuning.optimizers import run_optuna_tuner


def main():
    prices = load_prices_csv('examples/XBT_USD_prices.csv', dedupe='first')
    # fixed cost model
    fixed_slippage_pct = 0.001
    fixed_fee_pct = 0.00075

    def strategy_factory(**params):
        # Expect params: short, long, size
        return MovingAverageCrossover(short=int(params['short']), long=int(params['long']), size=float(params.get('size', 1.0)))

    def build_simulator():
        return Simulator(fixed_slippage_pct=fixed_slippage_pct, fixed_fee_pct=fixed_fee_pct)

    # objective for Optuna: receives dict of sampled params (from run_optuna_tuner spec)
    def objective(params: Dict[str, Any]) -> float:
        # ensure valid short < long
        short = int(params['short'])
        long = int(params['long'])
        if short >= long:
            return float('-inf')
        # create a strategy factory wrapper for evaluate_walk_forward
        def strat_factory(**p):
            return MovingAverageCrossover(short=short, long=long, size=float(params.get('size', 1.0)))

        # Use two seeds to reduce variance
        seeds = [0, 1]
        all_seed_results = []
        for s in seeds:
            sim_factory = lambda seed=s, fs=fixed_slippage_pct, ff=fixed_fee_pct: Simulator(fixed_slippage_pct=fs, fixed_fee_pct=ff, seed=seed)
            res = evaluate_walk_forward(prices=prices, targets=None, simulator=sim_factory, window=120, step=30, strategy_factory=strat_factory)
            folds = res.get('folds', [])
            sharpes = [f.get('metrics', {}).get('sharpe', 0.0) for f in folds]
            mean_sharpe = float(sum(sharpes) / len(sharpes)) if sharpes else 0.0
            all_seed_results.append(mean_sharpe)

        # objective: mean across seeds
        agg = float(sum(all_seed_results) / len(all_seed_results)) if all_seed_results else float('-inf')
        return agg

    # define Optuna search_space spec accepted by run_optuna_tuner
    # include volatility filter params in the search space
    search_space = {
        'short': ('int', 3, 20),
        'long': ('int', 30, 200),
        'size': ('uniform', 0.1, 2.0),
        'vol_filter': ('categorical', [False, True]),
        'vol_window': ('int', 10, 60),
        'vol_threshold': ('uniform', 0.001, 0.02),
    }

    # store study in sqlite so it can be resumed/inspected
    os.makedirs('experiments/optuna_studies', exist_ok=True)
    db_path = os.path.join('experiments', 'optuna_studies', 'ma_study.db')
    storage_url = f'sqlite:///{db_path}'

    study = run_optuna_tuner(objective, search_space, n_trials=100, seed=42, storage=storage_url)

    summary = {'best_params': study.best_params if hasattr(study, 'best_params') else {}, 'best_value': float(study.best_value) if hasattr(study, 'best_value') else None, 'storage': storage_url}
    out_dir = os.path.join('experiments', 'artifacts', f'optuna_ma_{uuid.uuid4()}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'optuna_study_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('Wrote', os.path.join(out_dir, 'optuna_study_summary.json'))


if __name__ == '__main__':
    main()
