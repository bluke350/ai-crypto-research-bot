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

    summarize = sub.add_parser("summarize", help="Summarize a single run directory and write a summary.json")
    summarize.add_argument("--run-dir", type=str, default=None)
    summarize.add_argument("--run-id", type=str, default=None)
    summarize.add_argument("--artifacts-root", type=str, default="experiments/artifacts")

    track_p = sub.add_parser("track", help="List recent experiment runs and key metrics")
    track_p.add_argument("--artifacts-root", type=str, default="experiments/artifacts")
    track_p.add_argument("--limit", type=int, default=20)
    track_p.add_argument("--sort-by", type=str, default="created_at")
    track_p.add_argument("--desc", action="store_true")
    track_p.add_argument("--export", type=str, choices=("csv", "tsv", "json"), default=None, help="Export listing to file")
    track_p.add_argument("--export-path", type=str, default=None, help="Path to write exported file")
    track_p.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="Compare two run ids or paths")

    serve = sub.add_parser("serve", help="Serve a small web UI for browsing experiment runs")
    serve.add_argument("--artifacts-root", type=str, default="experiments/artifacts")
    serve.add_argument("--host", type=str, default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)

    paper_run = sub.add_parser("paper-run", help="Run paper-mode runner using a checkpoint and CSV/prices")
    paper_run.add_argument("--ckpt", required=True, help="Path to checkpoint file (pickle)")
    paper_run.add_argument("--prices-csv", default=None, help="Optional prices CSV to use")
    paper_run.add_argument("--out-root", default="experiments/artifacts")
    paper_run.add_argument("--pair", default="XBT/USD")
    paper_run.add_argument("--sim-seed", type=int, default=None)

    paper_live = sub.add_parser("paper-live", help="Run paper-mode live streamer using a checkpoint and a CSV/stream source")
    paper_live.add_argument("--ckpt", required=True, help="Path to checkpoint file (pickle)")
    paper_live.add_argument("--prices-csv", default=None, help="Path to CSV used as streaming source")
    paper_live.add_argument("--out-root", default="experiments/artifacts")
    paper_live.add_argument("--pair", default="XBT/USD")
    paper_live.add_argument("--sim-seed", type=int, default=None)
    paper_live.add_argument("--max-ticks", type=int, default=None, help="Limit number of ticks to process (for tests)")
    paper_live.add_argument("--stream-delay", type=float, default=0.0, help="Seconds to wait between ticks (0 for no sleep)")
    paper_live.add_argument("--sleep-between", action='store_true', help="Sleep between ticks to simulate real time")

    replay_p = sub.add_parser("replay", help="Replay a previous run from a repro.json (deterministic)")
    replay_p.add_argument("--repro-path", type=str, default=None, help="Path to repro.json produced by a run")
    replay_p.add_argument("--run-id", type=str, default=None, help="Run id under artifacts root to replay")
    replay_p.add_argument("--artifacts-root", type=str, default="experiments/artifacts")

    args = p.parse_args()
    # add: track subcommand is handled below by calling orchestration.track
    # --- summarize subcommand: inspect tuning/experiment artifacts ---
    if args.command == "summarize":
        run_dir = getattr(args, "run_dir", None)
        run_id = getattr(args, "run_id", None)
        artifacts_root = getattr(args, "artifacts_root", "experiments/artifacts")
        if run_dir is None and run_id is None:
            raise ValueError("either --run-dir or --run-id must be provided")
        if run_dir is None:
            run_dir = os.path.join(str(artifacts_root), str(run_id))
        if not os.path.exists(run_dir):
            raise FileNotFoundError(f"run dir not found: {run_dir}")
        # try to find results.json
        res_path = os.path.join(run_dir, "results.json")
        if not os.path.exists(res_path):
            # fallback: pick first json file under run_dir
            jfiles = [f for f in os.listdir(run_dir) if f.endswith('.json')]
            if not jfiles:
                raise FileNotFoundError(f"no results JSON found in {run_dir}")
            res_path = os.path.join(run_dir, jfiles[0])
        with open(res_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)

        def extract_best(d):
            # common patterns
            if isinstance(d, dict):
                if 'best_params' in d:
                    return d.get('best_params'), d.get('best_score') or d.get('best_metrics')
                if 'best' in d:
                    b = d['best']
                    if isinstance(b, dict):
                        params = b.get('params') or b.get('best_params') or b
                        metrics = b.get('metrics') or b.get('score') or None
                        return params, metrics
                if 'trials' in d and isinstance(d['trials'], list):
                    trials = d['trials']
                    # try to pick the trial with minimum score/objective
                    best_t = None
                    best_s = None
                    for t in trials:
                        if not isinstance(t, dict):
                            continue
                        for k in ('score', 'objective', 'val_score', 'metric'):
                            if k in t:
                                try:
                                    s = float(t[k])
                                except Exception:
                                    s = None
                                if s is not None and (best_s is None or s < best_s):
                                    best_s = s
                                    best_t = t
                    if best_t is not None:
                        params = best_t.get('params') or best_t.get('parameters') or best_t
                        metrics = {k: best_t.get(k) for k in ('score', 'objective', 'val_score', 'metric') if k in best_t}
                        return params, metrics

            # recursive search for dicts with 'params'
            def walk(obj):
                if isinstance(obj, dict):
                    if 'params' in obj and isinstance(obj['params'], dict):
                        return obj['params'], obj.get('metrics') or obj.get('score')
                    for v in obj.values():
                        r = walk(v)
                        if r is not None:
                            return r
                elif isinstance(obj, list):
                    for item in obj:
                        r = walk(item)
                        if r is not None:
                            return r
                return None

            found = walk(d)
            if found:
                return found
            return None, None

        params, metrics = extract_best(data)
        summary = {"run_dir": os.path.abspath(run_dir), "results_file": os.path.abspath(res_path), "best_params": params, "best_metrics": metrics}
        out_path = os.path.join(run_dir, "summary.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print("Summary written to:", out_path)
        print("Best params:")
        print(json.dumps(params, indent=2))
        print("Best metrics:")
        print(json.dumps(metrics, indent=2))
        return
    if args.command == "track":
        # defer to tracker utility
        try:
            from orchestration import track as _track
        except Exception:
            import importlib

            _track = importlib.import_module("orchestration.track")

        # if compare given, pass through; otherwise list and optionally export
        if getattr(args, "compare", None):
            # orchestration.track handles compare and export together
            cmd = []
            # call track.main-like behavior via module
            # emulate CLI args
            import sys

            sys.argv = ["track"]
            if args.artifacts_root:
                sys.argv += ["--artifacts-root", args.artifacts_root]
            if args.compare:
                sys.argv += ["--compare"] + list(args.compare)
            if args.export:
                sys.argv += ["--export", args.export]
            if args.export_path:
                sys.argv += ["--export-path", args.export_path]
            if args.limit:
                sys.argv += ["--limit", str(args.limit)]
            if args.sort_by:
                sys.argv += ["--sort-by", args.sort_by]
            if args.desc:
                sys.argv += ["--desc"]
            _track.main()
            return

        # normal listing / export
        runs = _track._collect_runs(getattr(args, "artifacts_root", "experiments/artifacts"))
        if not runs:
            print("no runs found")
            return
        # sort and slice locally to present
        key = args.sort_by
        try:
            runs = sorted(runs, key=lambda r: float(r.get(key)) if r.get(key) is not None else r.get("created_at") or "", reverse=bool(args.desc))
        except Exception:
            runs = runs
        limit = getattr(args, "limit", 20)
        for r in runs[:limit]:
            print("-" * 60)
            print("run_id:", r.get("run_id"))
            print("run_dir:", r.get("run_dir"))
            print("created_at:", r.get("created_at"))
            print("max_drawdown:", r.get("max_drawdown"))
            print("sharpe:", r.get("sharpe"))

        if args.export and args.export_path:
            _track.export_runs(runs[:limit], args.export_path, fmt=args.export)
        return
    if args.command == "serve":
        try:
            from orchestration.web import create_app
        except Exception as e:
            raise RuntimeError("Flask is required to run the web UI. Install it first.") from e
        app = create_app(getattr(args, "artifacts_root", "experiments/artifacts"))
        app.run(host=args.host, port=args.port)
        return
    if args.command == "paper-run":
        try:
            from orchestration.paper_runner import run_paper
        except Exception:
            from orchestration import paper_runner as _pr

            run_paper = _pr.run_paper
        out_dir = run_paper(args.ckpt, prices_csv=args.prices_csv, out_root=args.out_root, pair=args.pair, sim_seed=args.sim_seed)
        print("paper run completed; artifacts at:", out_dir)
        return
    if args.command == "paper-live":
        try:
            from orchestration.paper_live import run_live
        except Exception:
            from orchestration import paper_live as _pl

            run_live = _pl.run_live
        out_dir = run_live(args.ckpt, prices_csv=args.prices_csv, out_root=args.out_root, pair=args.pair, sim_seed=args.sim_seed, max_ticks=args.max_ticks, stream_delay=args.stream_delay, sleep_between=args.sleep_between)
        print("paper live run completed; artifacts at:", out_dir)
        return
    if args.command == "replay":
        # locate repro.json
        repro_path = getattr(args, "repro_path", None)
        if repro_path is None and getattr(args, "run_id", None):
            repro_path = os.path.join(getattr(args, "artifacts_root", "experiments/artifacts"), args.run_id, "repro.json")
        if repro_path is None or not os.path.exists(repro_path):
            raise FileNotFoundError(f"repro.json not found at: {repro_path}")
        with open(repro_path, 'r', encoding='utf-8') as fh:
            repro = json.load(fh)

        # repro contains 'args' saved from auto_run; convert to CLI flags
        saved_args = repro.get('args') or {}
        # build command to call auto_run under the current python executable
        import subprocess, sys

        cmd = [sys.executable, "-m", "orchestration.auto_run"]
        for k, v in saved_args.items():
            if v is None:
                continue
            flag = "--" + k.replace('_', '-')
            # boolean flags were stored as True/False
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
                continue
            # lists are not expected; convert other types to string
            cmd.append(flag)
            cmd.append(str(v))

        print("Replaying with command:", " ".join(cmd))
        # run subprocess and stream output
        subprocess.run(cmd, check=True)
        return
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
            # ensure index has no name to avoid ambiguous 'timestamp' column/index downstream
            prices.index.name = None
        else:
            idx = pd.date_range("2020-01-01", periods=120, freq="T")
            import numpy as np

            rng = np.random.default_rng(0)
            returns = rng.normal(scale=0.001, size=len(idx))
            price = 100.0 + np.cumsum(returns)
            prices = pd.DataFrame({"close": price}, index=idx)
            prices.index.name = None

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
