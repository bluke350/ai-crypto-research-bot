"""Orchestrate ensemble evaluations per checkpoint.

For each checkpoint this script will:
 - run N per-run stochastic evaluations (each with a different RNG seed) and save per-run CSVs
 - compute the averaged NAV across runs and save averaged CSV
 - print basic metrics (initial, final, return, max drawdown)

Usage:
  python tooling/ensemble_eval.py --ckpt models/ppo_sim_2k.pth --pair "XBT/USD" --data-root data/raw --out-dir results/ensembles --ensembles 5 10 20 --runs-per-ckpt 3 --base-seed 0
"""
from __future__ import annotations

import sys
import argparse
import os
import math
import numpy as np
import pandas as pd
import torch

# ensure repo root is on sys.path so `src` imports work when run as a script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.rl.evaluate_policy import run_eval


def max_drawdown(series: pd.Series) -> float:
    cummax = series.cummax()
    dd = (series - cummax) / cummax
    return float(dd.min())


def per_run_eval(ckpt: str, pair: str, data_root: str, out_dir: str, run_seed: int, action_scale: float, action_mode: str):
    """Run a single stochastic evaluation with a fixed seed and write CSV."""
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    out = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(ckpt))[0]}_run_{run_seed}.csv")
    # call run_eval with stochastic_runs=1 so it will sample once
    run_eval(ckpt, data_root, pair, out, stochastic_runs=1, action_scale=action_scale, action_mode=action_mode)
    return out


def averaged_from_runs(run_files, out_path):
    dfs = [pd.read_csv(p, parse_dates=["timestamp"]) for p in run_files]
    # align by timestamp (inner join)
    merged = pd.concat([df.set_index("timestamp")["nav"] for df in dfs], axis=1)
    mean_nav = merged.mean(axis=1)
    out_df = mean_nav.reset_index().rename(columns={0: "nav"})
    out_df.columns = ["timestamp", "nav"]
    out_df.to_csv(out_path, index=False)
    return out_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pair", required=True)
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--out-dir", default="results/ensembles")
    p.add_argument("--ensembles", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--runs-per-ckpt", type=int, default=3, help="Number of independent checkpoints to evaluate (different seeds used for training externally)")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--action-scale", type=float, default=5.0)
    p.add_argument("--action-mode", type=str, default="absolute", choices=("absolute", "relative"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for run_idx in range(args.runs_per_ckpt):
        # use seeds spaced out to diversify sampling
        train_seed = args.base_seed + run_idx
        ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
        ckpt_path = args.ckpt

        for ens in args.ensembles:
            # per-run files
            run_files = []
            for r in range(ens):
                seed = args.base_seed + run_idx * 100 + r
                out_run = os.path.join(args.out_dir, f"{ckpt_name}_seed{train_seed}_ens{ens}_run{r}.csv")
                np.random.seed(seed)
                torch.manual_seed(seed)
                # run single-run eval
                run_eval(ckpt_path, args.data_root, args.pair, out_run, stochastic_runs=1, action_scale=args.action_scale, action_mode=args.action_mode)
                run_files.append(out_run)

            # averaged CSV
            out_avg = os.path.join(args.out_dir, f"{ckpt_name}_seed{train_seed}_ens{ens}_avg.csv")
            df_avg = averaged_from_runs(run_files, out_avg)
            # compute metrics
            initial = float(df_avg["nav"].iloc[0])
            final = float(df_avg["nav"].iloc[-1])
            final_return = (final / initial - 1.0) * 100.0
            mdd = max_drawdown(df_avg["nav"]) * 100.0
            results.append({"ckpt": ckpt_path, "train_seed": train_seed, "ens": ens, "avg_csv": out_avg, "final_return": final_return, "max_drawdown": mdd})
            print(f"ckpt={ckpt_path} seed={train_seed} ens={ens} final_return={final_return:.4f}% mdd={mdd:.4f}% avg_csv={out_avg}")

    # write summary
    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    print("wrote summary to", os.path.join(args.out_dir, "summary.csv"))


if __name__ == "__main__":
    main()
