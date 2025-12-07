"""
Bot service orchestrator

This script scans available USD pairs in `data/raw`, extracts CSVs from parquet snapshots,
runs the existing controller/tuning pipelines (via the repo CLI modules), ranks candidates,
optionally trains/promotes top-K models, and can start a paper-trading adapter.

Safety: by default this script runs in dry-run mode. Use `--execute` to allow training/promotions,
and `--trade` plus `--live-confirm` to enable trading. Always rotate API keys before enabling live trade.

Usage examples:
  # dry-run (discover + tune only)
  .\.venv\Scripts\python.exe -m scripts.bot_service --once --n-trials 3 --top-k 1

  # full run: tune, train, promote top-1, then start paper-trade (requires explicit --execute)
  .\.venv\Scripts\python.exe -m scripts.bot_service --once --n-trials 10 --top-k 1 --execute --trade paper

This file is intentionally conservative: it shells out to existing repo entrypoints
(`scripts.continuous_controller`, `orchestration.pipelines.training_pipeline`, `scripts.promote_candidate`) so
behaviour matches your existing pipeline flags.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
EXAMPLES = ROOT / "examples"
ARTIFACTS = ROOT / "experiments" / "artifacts"
PROMOTIONS_FILE = ROOT / "experiments" / "promotions.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bot_service")


def discover_usd_pairs(data_root: Path) -> List[str]:
    pairs = set()
    for p in data_root.glob("**/bars.parquet"):
        parts = p.parts
        # prefer explicit two-part directories (e.g., XBT/USD)
        found = False
        for i in range(len(parts) - 1):
            a = parts[i].upper()
            b = parts[i + 1].upper()
            if b == "USD" and a not in ("DATA", "RAW"):
                pairs.add(f"{parts[i]}/{parts[i+1]}")
                found = True
                break
        if found:
            continue
        # fallback: convert single tokens like XBTUSD -> XBT/USD
        for i, part in enumerate(parts):
            up = part.upper()
            if up == "USD":
                continue
            if up.endswith("USD") and len(up) > 3:
                base = part[:-3]
                if base:
                    pairs.add(f"{base}/{up[-3:]}")
                    break
    return sorted(pairs)


def parquet_to_csv_for_pair(pair: str, out_dir: Path) -> Path:
    candidate_parquets = []
    for p in DATA_RAW.glob("**/bars.parquet"):
        if pair.replace("/", "") in str(p):
            candidate_parquets.append(p)
    if not candidate_parquets:
        for p in DATA_RAW.glob("**/bars.parquet"):
            if pair.split("/")[0] in str(p) and "USD" in str(p).upper():
                candidate_parquets.append(p)
    if not candidate_parquets:
        raise FileNotFoundError(f"No parquet found for pair {pair}")
    latest = max(candidate_parquets, key=lambda p: p.stat().st_mtime)
    df = pd.read_parquet(latest)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        df = df.reset_index()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{pair.replace('/', '_')}_prices.csv"
    df.to_csv(out, index=False)
    logger.info("Wrote CSV for %s -> %s", pair, out)
    return out


def run_continuous_controller(csv_path: Path, pair: str, n_trials: int, parallel: int, dry_run: bool) -> Tuple[int, str]:
    cmd = [sys.executable, "-m", "scripts.continuous_controller", "--once", "--parallel", str(parallel), "--n-trials", str(n_trials), "--prices-csv", str(csv_path), "--symbols", pair]
    if dry_run:
        cmd.append("--dry-run")
    logger.info("Running tuning: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    logger.debug(proc.stdout)
    if proc.returncode != 0:
        logger.warning("Controller returned non-zero (%s) for %s. stderr:\n%s", proc.returncode, pair, proc.stderr)
    return proc.returncode, proc.stdout + "\n" + proc.stderr


def find_latest_artifact_metrics() -> Dict[str, float]:
    results: Dict[str, float] = {}
    if not ARTIFACTS.exists():
        return results
    for d in sorted(ARTIFACTS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        for j in d.glob("**/*.json"):
            try:
                text = j.read_text(encoding="utf-8")
                data = json.loads(text)
                if isinstance(data, dict):
                    for k in ("sharpe", "mean_sharpe", "mean_score", "score"):
                        if k in data:
                            pair = d.name
                            results[pair] = float(data[k])
                            logger.info("Found metric %s=%s in %s", k, data[k], j)
            except Exception:
                continue
    return results


def train_and_promote(pair: str, steps: int, register: bool) -> None:
    save_name = f"models/model_{pair.replace('/', '-')}.pth"
    train_cmd = [sys.executable, "-m", "orchestration.pipelines.training_pipeline", "--model", "rl", "--steps", str(steps), "--save", str(save_name), "--replay-pair", pair, "--data-root", str(DATA_RAW), "--out", str(ARTIFACTS)]
    if register:
        train_cmd.append("--register")
    logger.info("Running training: %s", " ".join(train_cmd))
    proc = subprocess.run(train_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("Training failed for %s: %s", pair, proc.stderr)
        return
    logger.info("Training finished for %s; saved: %s", pair, save_name)

    promote_cmd = [sys.executable, "-m", "scripts.promote_candidate", str(ARTIFACTS), "--checkpoint", str(save_name), "--models-dir", str(ROOT / "models"), "--promotions-file", str(PROMOTIONS_FILE)]
    # promote_candidate defaults to registering; only pass --no-register when register is False
    if not register:
        promote_cmd.append("--no-register")
    logger.info("Promoting: %s", " ".join(promote_cmd))
    proc2 = subprocess.run(promote_cmd, capture_output=True, text=True)
    if proc2.returncode != 0:
        logger.error("Promotion failed for %s: %s", pair, proc2.stderr)
    else:
        logger.info("Promotion succeeded for %s", pair)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="bot_service")
    p.add_argument("--once", action="store_true", help="Run once then exit")
    p.add_argument("--loop-interval", type=int, default=60, help="Minutes between runs when not --once")
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--steps", type=int, default=2000, help="Training steps for full train stage")
    p.add_argument("--execute", action="store_true", help="Allow training/promote/trade actions")
    p.add_argument("--trade", choices=("disabled", "paper", "live"), default="disabled")
    p.add_argument("--pairs", nargs="*", help="Optional explicit list of pairs to consider")
    p.add_argument("--n-candidates", type=int, default=10)
    args = p.parse_args(argv)

    if args.trade == "live" and not args.execute:
        logger.error("Live trading requires --execute. Aborting.")
        return 2

    logger.info("Bot starting; execute=%s trade=%s", args.execute, args.trade)
    def run_iteration() -> int:
        selector_path = ROOT / "experiments" / "pair_selection.json"
        pairs: List[str] = []
        pair_scores: Dict[str, float] = {}
        if args.pairs:
            pairs = args.pairs
        elif selector_path.exists():
            try:
                sel = json.loads(selector_path.read_text(encoding="utf-8"))
                for c in sel.get("candidates", []):
                    p = c.get("pair")
                    if p:
                        pairs.append(p)
                        pair_scores[p] = float(c.get("score", 0.0))
            except Exception:
                logger.exception("failed to read pair selector output; falling back to parquet discovery")
                pairs = discover_usd_pairs(DATA_RAW)
        else:
            try:
                # request at least `top_k` candidates from the selector
                requested = max(args.n_candidates, args.top_k)
                run_cmd = [sys.executable, "-m", "scripts.pair_selector", "--top-k", str(requested), "--lookback-minutes", str(60 * 24)]
                logger.info("Running pair selector: %s", " ".join(run_cmd))
                subprocess.run(run_cmd, check=True)
                sel = json.loads(selector_path.read_text(encoding="utf-8"))
                for c in sel.get("candidates", []):
                    p = c.get("pair")
                    if p:
                        pairs.append(p)
                        pair_scores[p] = float(c.get("score", 0.0))
            except Exception:
                logger.exception("pair selector failed; falling back to parquet discovery")
                pairs = discover_usd_pairs(DATA_RAW)

        # ensure we have at least `top_k` candidates; if selector/discovery returned fewer,
        # extend using raw parquet discovery (preserving existing ordering and uniqueness)
        if len(pairs) < args.top_k:
            found = discover_usd_pairs(DATA_RAW)
            for p in found:
                if p not in pairs:
                    pairs.append(p)
                if len(pairs) >= args.top_k:
                    break

        if not pairs:
            logger.error("No USD pairs discovered under %s", DATA_RAW)
            return 1
        logger.info("Candidate pairs (pre-ranked): %s", pairs)

        csvs: Dict[str, Path] = {}
        for pair in pairs:
            try:
                out = parquet_to_csv_for_pair(pair, EXAMPLES)
                csvs[pair] = out
            except Exception as e:
                logger.warning("Skipping %s: %s", pair, e)

        # Ensure we have numeric selector scores for all candidate pairs.
        metrics: Dict[str, float] = dict(pair_scores)
        if pairs:
            try:
                # try to import score_pair for fast in-process scoring
                from scripts.pair_selector import score_pair as _score_pair
            except Exception:
                _score_pair = None

            for p in pairs:
                if p not in metrics:
                    try:
                        if _score_pair is not None:
                            s = _score_pair(p, lookback_minutes=60 * 24)
                            metrics[p] = float(s.get("score", 0.0))
                            pair_scores.setdefault(p, float(s.get("score", 0.0)))
                        else:
                            # fallback: call pair_selector as a CLI to get score for this pair
                            cmd = [sys.executable, "-m", "scripts.pair_selector", "--top-k", "1", "--lookback-minutes", str(60 * 24)]
                            subprocess.run(cmd, check=False)
                            # after CLI run, try reading selector file again
                            try:
                                sel = json.loads((ROOT / "experiments" / "pair_selection.json").read_text(encoding="utf-8"))
                                for c in sel.get("candidates", []):
                                    if c.get("pair") == p:
                                        metrics[p] = float(c.get("score", 0.0))
                                        pair_scores.setdefault(p, float(c.get("score", 0.0)))
                                        break
                            except Exception:
                                metrics[p] = 0.0
                                pair_scores.setdefault(p, 0.0)
                    except Exception:
                        metrics[p] = 0.0
        for pair, csv in csvs.items():
            rc, output = run_continuous_controller(csv, pair, args.n_trials, args.parallel, dry_run=not args.execute)
            logger.debug(output)
            try:
                for line in (output or "").splitlines():
                    if "sharpe" in line.lower():
                        import re

                        m = re.search(r"sharpe\s*[:=]\s*([-+]?[0-9]*\.?[0-9]+)", line, re.IGNORECASE)
                        if m:
                            metrics[pair] = float(m.group(1))
                            break
            except Exception:
                pass

        # ensure every candidate pair has a metric (fallback to selector score or 0.0)
        for p in pairs:
            metrics.setdefault(p, float(pair_scores.get(p, 0.0)))

        if not metrics:
            logger.warning("No metrics available after tuning; exiting iteration")
            return 0

        ranked = sorted(metrics.items(), key=lambda kv: kv[1], reverse=True)
        logger.info("Ranked candidates: %s", ranked[: args.top_k])

        if args.execute:
            for pair, score in ranked[: args.top_k]:
                logger.info("Training/promoting %s (score=%s)", pair, score)
                train_and_promote(pair, args.steps, register=True)

        if args.trade != "disabled":
            if not args.execute:
                logger.warning("Trade requested but --execute not set; skipping trade")
            else:
                try:
                    from orchestration.paper_live import run_live
                except Exception as e:
                    logger.exception("Failed to import orchestration.paper_live: %s", e)
                    return 1

                for pair, _ in ranked[: args.top_k]:
                    logger.info("Starting paper_live for %s (top candidate)", pair)
                    promoted = None
                    models_dir = ROOT / "models"
                    for f in sorted(models_dir.glob("promoted_*"), key=lambda p: p.stat().st_mtime, reverse=True):
                        if pair.replace('/', '-') in f.name or pair.replace('/', '') in f.name:
                            promoted = str(f)
                            break
                    if promoted is None:
                        for f in sorted(models_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True):
                            if pair.replace('/', '-') in f.name or pair.replace('/', '') in f.name:
                                promoted = str(f)
                                break

                    prices_csv = csvs.get(pair)
                    if prices_csv is None:
                        logger.warning("No CSV prices for %s; skipping paper run", pair)
                        continue

                    try:
                        run_root = str(ARTIFACTS)
                        run_live(checkpoint_path=promoted or '', prices_csv=str(prices_csv), out_root=run_root, pair=pair, paper_mode=True)
                        logger.info("paper_live completed for %s", pair)
                    except Exception as e:
                        logger.exception("paper_live run failed for %s: %s", pair, e)

        return 0

    try:
        if args.once:
            return run_iteration()
        else:
            logger.info("Entering continuous loop; interval=%s minutes", args.loop_interval)
            while True:
                run_iteration()
                logger.info("Iteration complete; sleeping %s minutes", args.loop_interval)
                import time

                time.sleep(args.loop_interval * 60)
    except KeyboardInterrupt:
        logger.info("Interrupted; exiting")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
