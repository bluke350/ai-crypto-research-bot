from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from statistics import mean
from typing import Dict, List, Optional


def find_usd_pairs(data_root: str) -> List[str]:
    pairs = set()
    # match data/raw/*/USD (folder structure like ETH/USD)
    pattern1 = os.path.join(data_root, "*", "USD")
    for p in glob.glob(pattern1):
        base = os.path.basename(os.path.dirname(p)) + "/USD"
        pairs.add(base)
    # match data/raw/*USD (folder structure like XBTUSD)
    pattern2 = os.path.join(data_root, "*USD")
    for p in glob.glob(pattern2):
        name = os.path.basename(p)
        # convert XBTUSD -> XBT/USD
        if name.endswith("USD") and len(name) > 3:
            base = name[:-3] + "/USD"
            pairs.add(base)
    return sorted(pairs)


def run_tuning_for_pair(pair: str, prices_csv: str, n_trials: int, out_root: str, register: bool) -> Optional[str]:
    run_id = f"auto_{pair.replace('/', '-') }"
    out_dir = os.path.join(out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)
    args = [sys.executable, "-m", "orchestration.pipelines.tuning_pipeline", "--symbol", pair, "--n-trials", str(n_trials), "--output", out_dir, "--prices-csv", prices_csv]
    if register:
        args.append("--register")
    print("Running tuning for", pair, "->", out_dir)
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print("Tuning failed for", pair, e)
        return None
    # find results.json
    for root, dirs, files in os.walk(out_dir):
        if "results.json" in files:
            return os.path.join(root, "results.json")
    return None


def parse_mean_sharpe(results_path: str) -> float:
    try:
        with open(results_path, "r", encoding="utf-8") as fh:
            r = json.load(fh)
    except Exception:
        return 0.0
    folds = r.get("folds") or []
    sharps = []
    for f in folds:
        m = f.get("metrics") or {}
        v = m.get("sharpe")
        if v is not None:
            try:
                sharps.append(float(v))
            except Exception:
                pass
    return float(mean(sharps)) if sharps else 0.0


def csv_from_parquet(pair: str, data_root: str, out_csv: Optional[str]) -> Optional[str]:
    # Use the tooling/parquet_to_csv.py helper script to extract a CSV from the first parquet
    script = os.path.join(os.path.dirname(__file__), "..", "tooling", "parquet_to_csv.py")
    script = os.path.abspath(script)
    out = out_csv or os.path.join("examples", pair.replace('/', '_') + "_prices.csv")
    args = [sys.executable, script, "--pair", pair, "--data-root", data_root, "--out", out]
    try:
        subprocess.run(args, check=True)
        return out
    except subprocess.CalledProcessError:
        print("Failed to create CSV for", pair)
        return None


def train_and_promote(pair: str, data_root: str, artifacts_root: str, save_dir: str, register: bool) -> bool:
    # train
    save_name = f"models/model_{pair.replace('/', '-')}.pth"
    train_args = [sys.executable, "-m", "orchestration.pipelines.training_pipeline", "--model", "rl", "--steps", "500", "--save", save_name, "--replay-pair", pair, "--data-root", data_root, "--out", artifacts_root]
    if register:
        train_args.append("--register")
    print("Training for", pair)
    try:
        subprocess.run(train_args, check=True)
    except subprocess.CalledProcessError as e:
        print("Training failed for", pair, e)
        return False

    # find artifact dir (last created under artifacts_root with prefix controller or run id)
    # For simplicity, attempt to find a recent artifact folder that was created
    cand = None
    if os.path.isdir(artifacts_root):
        entries = sorted(os.listdir(artifacts_root), key=lambda x: os.path.getmtime(os.path.join(artifacts_root, x)), reverse=True)
        for e in entries:
            p = os.path.join(artifacts_root, e)
            if os.path.isdir(p):
                cand = p
                break

    ckpt = os.path.abspath(save_name)
    # promote (register)
    promo_args = [sys.executable, "-m", "scripts.promote_candidate", cand or '.', "--checkpoint", ckpt, "--models-dir", save_dir, "--promotions-file", os.path.join(artifacts_root, "promotions.json")]
    if not register:
        promo_args.append("--no-register")
    try:
        subprocess.run(promo_args, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print("Promotion failed for", pair, e)
        return False


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--artifacts-root", default="experiments/artifacts")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--pairs", nargs="*", default=None, help="Optional list of pairs to consider (format: XBT/USD)")
    p.add_argument("--execute", action="store_true", help="If present, run training and promote. Otherwise dry-run tuning only.")
    p.add_argument("--prices-out-dir", default="examples")
    args = p.parse_args(argv)

    os.makedirs(args.artifacts_root, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.prices_out_dir, exist_ok=True)

    pairs = args.pairs or find_usd_pairs(args.data_root)
    if not pairs:
        print("No USD pairs found under", args.data_root)
        return 1

    print("Pairs to evaluate:", pairs)

    results: Dict[str, float] = {}
    results_json: Dict[str, str] = {}

    for pair in pairs:
        csv_path = csv_from_parquet(pair, args.data_root, os.path.join(args.prices_out_dir, f"{pair.replace('/', '_')}_prices.csv"))
        if not csv_path:
            continue
        res = run_tuning_for_pair(pair, csv_path, args.n_trials, args.artifacts_root, register=args.execute)
        if not res:
            continue
        sharpe = parse_mean_sharpe(res)
        results[pair] = sharpe
        results_json[pair] = res
        print(f"Pair {pair} -> mean sharpe {sharpe}")

    if not results:
        print("No successful tuning runs")
        return 2

    # pick top-k
    ranked = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
    print("Ranked pairs:")
    for r in ranked:
        print(r)

    to_train = [p for p, s in ranked[: args.top_k]]

    if args.execute:
        for pair in to_train:
            ok = train_and_promote(pair, args.data_root, args.artifacts_root, args.models_dir, register=True)
            print(f"Train+promote {pair} -> {ok}")

    print("Auto controller completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
