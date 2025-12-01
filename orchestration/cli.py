from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import pandas as pd

from src.training.trainer import train_ml
from src.training.inference import ModelWrapper
from src.training.adapter import inference_backtest


def main():
    p = argparse.ArgumentParser(prog="orchestration-cli")
    sub = p.add_subparsers(dest="command")

    run_eval = sub.add_parser("eval", help="Train (optional) and run quick eval backtest using ML checkpoint")
    run_eval.add_argument("--train", action="store_true", help="Run training first to produce a checkpoint")
    run_eval.add_argument("--ckpt", type=str, default="models/ml_eval_ckpt.pkl")
    run_eval.add_argument("--data-root", type=str, default="data/raw")
    run_eval.add_argument("--method", type=str, default="vol_norm", choices=("threshold", "vol_norm"))
    run_eval.add_argument("--scale", type=float, default=1.0)
    run_eval.add_argument("--vol-window", type=int, default=20)
    run_eval.add_argument("--max-size", type=float, default=2.0)
    run_eval.add_argument("--out", type=str, default="experiments/eval")
    run_eval.add_argument("--prices-csv", type=str, default=None, help="Path to historical prices CSV with timestamp and 'close' columns")
    run_eval.add_argument("--timestamp-col", type=str, default="timestamp", help="Name of the timestamp column in CSV (default: 'timestamp')")
    run_eval.add_argument("--pair", type=str, default="XBT/USD", help="Trading pair to use for the ExchangeSimulator (default: XBT/USD)")

    args = p.parse_args()
    if args.command == "eval":
        Path(args.out).mkdir(parents=True, exist_ok=True)
        ckpt = args.ckpt
        if args.train:
            print("Running ML trainer to produce checkpoint...")
            train_ml(data_root=args.data_root, save=ckpt, steps=100, seed=0)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")

        # Load prices: either from provided CSV, or fall back to a small synthetic series
        if args.prices_csv:
            if not os.path.exists(args.prices_csv):
                raise FileNotFoundError(f"prices CSV not found: {args.prices_csv}")
            # load CSV; try to parse timestamp column or assume first column contains timestamps
            df = pd.read_csv(args.prices_csv)
            # prefer user-specified timestamp column
            if args.timestamp_col in df.columns:
                df[args.timestamp_col] = pd.to_datetime(df[args.timestamp_col])
                df = df.set_index(args.timestamp_col)
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
            else:
                # try to parse the first column as timestamp
                first_col = df.columns[0]
                try:
                    df[first_col] = pd.to_datetime(df[first_col])
                    df = df.set_index(first_col)
                except Exception:
                    raise ValueError("Could not find or parse a timestamp column in the CSV")
            if "close" not in df.columns:
                raise ValueError("prices CSV must contain a 'close' column")
            prices = pd.DataFrame({"close": df["close"].astype(float)}, index=df.index)
        else:
            idx = pd.date_range("2020-01-01", periods=120, freq="T")
            import numpy as np

            rng = np.random.default_rng(0)
            returns = rng.normal(scale=0.001, size=len(idx))
            price = 100.0 + np.cumsum(returns)
            prices = pd.DataFrame({"close": price}, index=idx)

        print("Loading checkpoint and running inference/backtest using ExchangeSimulator...")
        from src.execution.simulators import ExchangeSimulator
        sim = ExchangeSimulator(pair=args.pair)
        result = inference_backtest(ckpt, prices, sim, method=args.method, scale=args.scale, vol_window=args.vol_window, max_size=args.max_size)
        # dump summary
        out_path = Path(args.out) / "eval_summary.json"
        summary = {"pnl_head": list(map(float, result["pnl"].head(5).tolist())), "n_executions": len(result["executions"])}
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh)
        print(f"Eval complete; summary written to {out_path}")


if __name__ == "__main__":
    main()
