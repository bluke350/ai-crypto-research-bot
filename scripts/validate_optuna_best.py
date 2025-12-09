#!/usr/bin/env python3
"""Validate optuna best params by running a cost-aware walk-forward and producing diagnostics.

Usage: python scripts/validate_optuna_best.py <optuna_summary_json> [<prices_csv>] 
"""
import sys
import os
import json
import uuid
from typing import Dict, Any

from src.utils.io import load_prices_csv
from src.execution.simulator import Simulator
from src.strategies.moving_average import MovingAverageCrossover
from src.validation.walk_forward import evaluate_walk_forward

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_optuna_best.py <optuna_summary_json> [<prices_csv>]")
        sys.exit(2)
    summary_path = sys.argv[1]
    prices_csv = sys.argv[2] if len(sys.argv) >= 3 else 'examples/XBT_USD_prices.csv'

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    best = summary.get('best_params', {})
    print('Best params:', best)

    prices = load_prices_csv(prices_csv, dedupe='first')

    # prepare explicit simulator cost params (same cost model used during tuning)
    fixed_slippage_pct = 0.001
    fixed_fee_pct = 0.00075

    # build strategy factory using best params
    def strategy_factory(**params):
        # merge best (optuna) params with any overrides passed
        cfg = dict(best)
        cfg.update(params or {})
        # ensure ints/float types
        cfg_parsed = {}
        if 'short' in cfg:
            cfg_parsed['short'] = int(cfg['short'])
        if 'long' in cfg:
            cfg_parsed['long'] = int(cfg['long'])
        if 'size' in cfg:
            cfg_parsed['size'] = float(cfg['size'])
        return MovingAverageCrossover(**cfg_parsed)

    seeds = [0,1]
    all_seed_results = []
    out_dir = os.path.join('experiments', 'artifacts', f'validate_optuna_{uuid.uuid4()}')
    os.makedirs(out_dir, exist_ok=True)

    for s in seeds:
        sim_factory = lambda seed=s, fs=fixed_slippage_pct, ff=fixed_fee_pct: Simulator(fixed_slippage_pct=fs, fixed_fee_pct=ff, seed=seed)
        res = evaluate_walk_forward(prices=prices, targets=None, simulator=sim_factory, window=120, step=30, strategy_factory=strategy_factory)
        all_seed_results.append(res)

    # save result
    summary_out = {'seeds': []}
    for idx, sres in enumerate(all_seed_results):
        # compute per-seed summary similar to pipeline
        folds = sres.get('folds', [])
        final_vals = [f.get('metrics', {}).get('final_value', 0.0) for f in folds]
        sharpes = [f.get('metrics', {}).get('sharpe', 0.0) for f in folds]
        seed_summary = {
            'seed': int(seeds[idx]),
            'n_folds': len(folds),
            'mean_final_value': float(sum(final_vals)/len(final_vals)) if final_vals else 0.0,
            'mean_sharpe': float(sum(sharpes)/len(sharpes)) if sharpes else 0.0,
            'folds': folds,
        }
        summary_out['seeds'].append(seed_summary)
    # aggregate
    mean_final_values = [sr['mean_final_value'] for sr in summary_out['seeds']]
    mean_sharpes = [sr['mean_sharpe'] for sr in summary_out['seeds']]
    summary_out['aggregate'] = {
        'n_seeds': len(summary_out['seeds']),
        'mean_final_value': float(sum(mean_final_values)/len(mean_final_values)) if mean_final_values else 0.0,
        'mean_sharpe': float(sum(mean_sharpes)/len(mean_sharpes)) if mean_sharpes else 0.0,
    }

    out_path = os.path.join(out_dir, 'results.json')
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(summary_out, fh, indent=2)
    print('Wrote', out_path)

    # generate diagnostics via the existing script if desired (attempt to import matplotlib/pandas)
    try:
        from scripts.generate_walkforward_diagnostics import main as gen_diag_main
        gen_diag_main()
    except Exception:
        # fallback: write a simple diagnostics summary
        diag = {'n_folds_total': sum(len(s['folds']) for s in summary_out['seeds']), 'aggregate': summary_out['aggregate']}
        with open(os.path.join(out_dir, 'diagnostics_summary.json'), 'w', encoding='utf-8') as fh:
            json.dump(diag, fh, indent=2)
        print('Wrote diagnostics_summary.json')


if __name__ == '__main__':
    main()
