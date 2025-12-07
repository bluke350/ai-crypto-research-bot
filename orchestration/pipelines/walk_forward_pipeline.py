from __future__ import annotations

import argparse
import uuid
import json
import os
import logging
from typing import Optional

import pandas as pd

from src.utils.config import load_yaml
from src.utils.io import load_prices_csv
from src.validation.walk_forward import evaluate_walk_forward
from src.execution.simulator import Simulator
from src.strategies.moving_average import MovingAverageCrossover
from src.ingestion.providers import kraken_rest

try:
    # optional adapter only required when evaluating RL policies
    from src.models.rl.policy_adapter import RLPolicyStrategy
except Exception:
    RLPolicyStrategy = None


LOG = logging.getLogger(__name__)





def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default=None)
    p.add_argument("--prices-csv", type=str, default=None)
    p.add_argument("--dedupe", type=str, default="first", choices=["none", "first", "last", "mean"], help="How to handle duplicate timestamps in CSV: none|first|last|mean")
    p.add_argument("--ppo-checkpoint", type=str, default=None, help="Path to PPO checkpoint to evaluate (optional)")
    p.add_argument("--obs-window", type=int, default=None, help="Observation window length for RL policy (optional). If omitted, inferred from checkpoint when possible.")
    p.add_argument("--window", type=int, default=120)
    p.add_argument("--step", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=str, default=None, help="Comma separated seeds to run, e.g. '0,1,2'")
    p.add_argument("--register", action="store_true", help="Register runs/artifacts in experiments DB")
    p.add_argument("--output", type=str, default="experiments/artifacts")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    run_id = str(uuid.uuid4())
    out_dir = os.path.join(args.output, run_id)
    os.makedirs(out_dir, exist_ok=True)

    cfg = {}
    try:
        cfg = load_yaml("configs/project.yaml").get("project", {})
    except Exception:
        # best effort; not fatal
        pass

    symbol = args.symbol
    if symbol is None:
        try:
            kcfg = load_yaml("configs/kraken.yaml").get("kraken", {})
            syms = kcfg.get("symbols", [])
            symbol = syms[0] if syms else "XBT/USD"
        except Exception:
            symbol = "XBT/USD"

    # load price history
    prices: Optional[pd.DataFrame] = None
    if args.prices_csv:
        prices = load_prices_csv(args.prices_csv, dedupe=args.dedupe)
    else:
        try:
            # attempt to fetch via Kraken REST (since=0 -> best-effort full history)
            prices = kraken_rest.get_ohlc(symbol, "1m", since=0)
        except Exception as e:
            LOG.warning("Failed to load OHLC from kraken REST: %s", e)
            raise

    # pick strategy factory
    if args.ppo_checkpoint:
        if RLPolicyStrategy is None:
            raise RuntimeError("RLPolicyStrategy adapter not available (torch required)")
        # create a factory that will be called by evaluate_walk_forward(**params)
        def strategy_factory(**params):
            # users may tune `window` or `size` via param_space
            # allow explicit obs_window from CLI, otherwise let adapter infer from checkpoint
            if getattr(args, "obs_window", None) is not None:
                params = dict(params)
                params["obs_window"] = int(args.obs_window)
            return RLPolicyStrategy(ckpt_path=args.ppo_checkpoint, **params)
    else:
        strategy_factory = MovingAverageCrossover

    # determine seeds to run
    if args.seeds:
        seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    else:
        seeds = [int(args.seed)]

    all_seed_results = []
    for s in seeds:
        # simulator factory per-seed
        sim_factory = lambda seed=s: Simulator(seed=seed)
        res = evaluate_walk_forward(prices=prices, targets=None, simulator=sim_factory, window=args.window, step=args.step, strategy_factory=strategy_factory)
        # aggregate fold-level metrics into per-seed summary
        folds = res.get("folds", [])
        final_vals = [f.get("metrics", {}).get("final_value", 0.0) for f in folds]
        sharpes = [f.get("metrics", {}).get("sharpe", 0.0) for f in folds]
        seed_summary = {
            "seed": int(s),
            "n_folds": len(folds),
            "mean_final_value": float(sum(final_vals) / len(final_vals)) if final_vals else 0.0,
            "mean_sharpe": float(sum(sharpes) / len(sharpes)) if sharpes else 0.0,
            "folds": folds,
        }
        all_seed_results.append(seed_summary)

    # aggregate across seeds
    mean_final_values = [sr["mean_final_value"] for sr in all_seed_results]
    mean_sharpes = [sr["mean_sharpe"] for sr in all_seed_results]
    agg = {
        "n_seeds": len(all_seed_results),
        "mean_final_value": float(sum(mean_final_values) / len(mean_final_values)) if mean_final_values else 0.0,
        "mean_sharpe": float(sum(mean_sharpes) / len(mean_sharpes)) if mean_sharpes else 0.0,
    }

    summary = {"seeds": all_seed_results, "aggregate": agg}

    # dump results
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # optionally register to experiments DB
    if getattr(args, "register", False):
        try:
            from src.persistence.db import RunLogger
            run_id2 = str(uuid.uuid4())
            with RunLogger(run_id2, cfg={"pipeline": "walk_forward", "seeds": seeds}):
                rl_path = os.path.join(out_dir, "results.json")
                try:
                    with RunLogger(run_id2, cfg={}) as rl:
                        rl.log_artifact(os.path.join(out_dir, "results.json"), kind='walk_forward')
                except Exception:
                    LOG.exception('failed to register walk-forward artifact')
        except Exception:
            LOG.exception('RunLogger not available; skipping registration')

    print(f"Walk-forward completed; artifacts in {out_dir}")


if __name__ == "__main__":
    main()
