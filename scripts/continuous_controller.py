from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import threading
from statistics import mean
from typing import List, Tuple, Dict, Any
import shutil
import tempfile
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from src.utils.config import load_yaml
from src.operations import safety
from src.persistence.db import RunLogger
from src.metrics.emitter import emit as emit_metric
from src.training.online_retrainer import OnlineRetrainer, RetrainPolicy, example_retrain_fn
from pathlib import Path
import time

LOG = logging.getLogger(__name__)


def run_module(module: str, args: List[str]) -> None:
    cmd = [sys.executable, "-m", module] + args
    LOG.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def find_results_json(output_root: str) -> str | None:
    for root, dirs, files in os.walk(output_root):
        if "results.json" in files:
            return os.path.join(root, "results.json")
    return None


def parse_mean_sharpe(results_path: str) -> float:
    with open(results_path, "r", encoding="utf-8") as fh:
        r = json.load(fh)
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


def _run_tuning_for_symbol(symbol: str, args: argparse.Namespace) -> Tuple[str, str, str, Dict[str, Any], str]:
    run_id = str(uuid.uuid4())
    out_dir = os.path.join(args.artifacts_root, f"controller_{run_id}_{symbol.replace('/', '-')}")
    os.makedirs(out_dir, exist_ok=True)

    tune_args = [
        "--symbol",
        symbol,
        "--n-trials",
        str(args.n_trials),
        "--output",
        out_dir,
    ]
    if args.prices_csv:
        tune_args.extend(["--prices-csv", args.prices_csv])
    if args.use_optuna:
        tune_args.extend(["--optimizer", "optuna", "--optuna-db", args.optuna_db])
    if args.execute:
        tune_args.append("--register")

    emit_metric("tuning_started", {"symbol": symbol, "run_id": run_id, "n_trials": args.n_trials})
    try:
        run_module("orchestration.pipelines.tuning_pipeline", tune_args)
    except subprocess.CalledProcessError as e:
        LOG.exception("tuning failed for %s: %s", symbol, e)
        emit_metric("tuning_failed", {"symbol": symbol, "run_id": run_id, "err": str(e)})
        return symbol, out_dir, "", {}, run_id

    results_path = find_results_json(out_dir) or ""
    cand_sharpe = parse_mean_sharpe(results_path) if results_path else 0.0
    candidate_metrics: Dict[str, Any] = {"sharpe": cand_sharpe}
    try:
        with open(results_path, 'r', encoding='utf-8') as fh:
            jr = json.load(fh)
        fvals = [f.get('metrics', {}).get('final_value') for f in jr.get('folds', []) if f.get('metrics', {}).get('final_value') is not None]
        if fvals:
            candidate_metrics['final_value'] = float(mean([float(x) for x in fvals]))
    except Exception:
        pass

    emit_metric("tuning_finished", {"symbol": symbol, "run_id": run_id, "metrics": candidate_metrics})
    return symbol, out_dir, results_path or "", candidate_metrics, run_id


def extract_best_candidate(results_path: str, out_dir: str) -> Dict[str, Any] | None:
    """Read a tuning `results.json`, pick the fold with highest `best_score`,
    write `BEST_CANDIDATE.json` into `out_dir` and return a dict with best params/metrics.
    Returns None if no candidate found.
    """
    try:
        with open(results_path, "r", encoding="utf-8") as fh:
            r = json.load(fh)
    except Exception:
        LOG.exception("failed to read results.json at %s", results_path)
        return None

    folds = r.get("folds") or []
    best = None
    best_score = float("-inf")
    for f in folds:
        sc = f.get("best_score") if isinstance(f.get("best_score"), (int, float)) else None
        if sc is None:
            sc = f.get("metrics", {}).get("sharpe")
        try:
            scf = float(sc) if sc is not None else float("-inf")
        except Exception:
            scf = float("-inf")
        if scf > best_score:
            best_score = scf
            best = f

    if best is None:
        return None

    candidate = {
        "run_id": os.path.basename(out_dir),
        "best_score": best_score,
        "best_params": best.get("best_params") or {},
        "metrics": best.get("metrics") or {},
    }

    try:
        dest = os.path.join(out_dir, "BEST_CANDIDATE.json")
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(candidate, fh, indent=2)
    except Exception:
        LOG.exception("failed to write BEST_CANDIDATE.json in %s", out_dir)

    return candidate


def train_candidate_from_csv(candidate: Dict[str, Any], out_dir: str, prices_csv: str, artifacts_root: str) -> None:
    """Run the ML training pipeline using the CSV provided and register the produced checkpoint.
    This creates a temporary data folder with the CSV and invokes the training pipeline as a module.
    """
    if not prices_csv or not os.path.exists(prices_csv):
        LOG.warning("no prices_csv available; skipping candidate training for %s", out_dir)
        return

    # create temp folder for training data and copy csv into it
    td = None
    try:
        td = tempfile.mkdtemp(prefix="controller_train_")
        dst = os.path.join(td, os.path.basename(prices_csv))
        shutil.copy2(prices_csv, dst)

        save_path = os.path.join(out_dir, "candidate_model.pkl")
        train_args = [
            "--model",
            "ml",
            "--steps",
            "200",
            "--save",
            save_path,
            "--data-root",
            td,
            "--out",
            artifacts_root,
            "--register",
        ]

        emit_metric("candidate_training_started", {"out_dir": out_dir, "save": save_path})
        try:
            run_module("orchestration.pipelines.training_pipeline", train_args)
            emit_metric("candidate_training_finished", {"out_dir": out_dir, "save": save_path})
        except subprocess.CalledProcessError:
            LOG.exception("candidate training failed for %s", out_dir)
            emit_metric("candidate_training_failed", {"out_dir": out_dir})

    finally:
        # cleanup temp data folder
        try:
            if td and os.path.exists(td):
                shutil.rmtree(td)
        except Exception:
            LOG.exception("failed to remove temp training dir %s", td)


def csv_to_replay_parquet(prices_csv: str, pair: str, data_root: str = "data/raw") -> str | None:
    """Convert a CSV of OHLCV rows into the repo's replay parquet layout:
    data_root/{PAIR}/{YYYYMMDD}/{YYYYMMDDTHHMM}.parquet
    Returns the path to the directory written for the pair or None on failure.
    """
    try:
        df = pd.read_csv(prices_csv)
    except Exception:
        LOG.exception("failed to read prices csv %s", prices_csv)
        return None

    # Ensure timestamp column exists and is datetime
    if "timestamp" not in df.columns:
        # try common alternatives
        for cand in ("time", "datetime", "date"):
            if cand in df.columns:
                df = df.rename(columns={cand: "timestamp"})
                break

    if "timestamp" not in df.columns:
        LOG.error("prices CSV missing timestamp column: %s", prices_csv)
        return None

    try:
        if not is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    except Exception:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Keep only expected columns
    expected = ["timestamp", "open", "high", "low", "close", "volume"]
    present = [c for c in expected if c in df.columns]
    out_df = df[present].copy()

    # create output dir per pair
    pair_dir = os.path.join(data_root, pair.replace("/", os.sep))
    os.makedirs(pair_dir, exist_ok=True)

    # write a single parquet file named by first timestamp (safe for small inputs)
    try:
        ts0 = out_df["timestamp"].iloc[0]
        fname = pd.Timestamp(ts0).strftime("%Y%m%dT%H%M") + ".parquet"
    except Exception:
        fname = "manual.parquet"

    # create date subfolder
    try:
        date_folder = pd.Timestamp(out_df["timestamp"].iloc[0]).strftime("%Y%m%d")
    except Exception:
        date_folder = "manual"

    write_dir = os.path.join(pair_dir, date_folder)
    os.makedirs(write_dir, exist_ok=True)
    out_path = os.path.join(write_dir, fname)
    try:
        out_df.to_parquet(out_path, index=False)
        LOG.info("wrote replay parquet for %s -> %s", pair, out_path)
        return pair_dir
    except Exception:
        LOG.exception("failed to write parquet to %s", out_path)
        return None


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true", help="Run one loop and exit")
    p.add_argument("--daemon", action="store_true", help="Run continuously as a daemon (loop and sleep)")
    p.add_argument("--interval", type=int, default=300, help="Interval seconds between daemon runs")
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--prices-csv", type=str, default=None)
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--execute", action="store_true", help="Allow controller to execute training (default: dry-run).")
    p.add_argument("--use-optuna", action="store_true", help="Enable Optuna optimizer and persistence")
    p.add_argument("--optuna-db", type=str, default="sqlite:///experiments/optuna.db")
    p.add_argument("--parallel", type=int, default=1, help="Number of parallel tuning workers")
    p.add_argument("--artifacts-root", type=str, default="experiments/artifacts")
    p.add_argument("--enable-retrainer", action="store_true", help="Enable online retrainer to trigger candidate retraining")
    p.add_argument("--retrain-metric", type=str, default="val_loss", help="Metric name to monitor for retraining")
    p.add_argument("--retrain-direction", choices=["down", "up"], default="down")
    p.add_argument("--retrain-threshold", type=float, default=0.02)
    p.add_argument("--retrain-window", type=int, default=5)
    p.add_argument("--retrain-cooldown", type=int, default=3600)
    p.add_argument("--retrain-mode", choices=["mean", "ewma"], default="ewma")
    p.add_argument("--retrain-ewma-alpha", type=float, default=0.3)
    p.add_argument("--retrain-db", type=str, default=None, help="Optional DB URL for retrainer RunLogger")
    args = p.parse_args(argv)

    os.makedirs(args.artifacts_root, exist_ok=True)

    stop_event = threading.Event()

    def _sig_handler(signum, frame):
        LOG.info("Received signal %s, stopping after current run", signum)
        stop_event.set()

    try:
        signal.signal(signal.SIGTERM, _sig_handler)
        signal.signal(signal.SIGINT, _sig_handler)
    except Exception:
        # signal may not be available on some platforms; ignore
        pass

    def run_once():
        # load symbols from config if not provided
        symbols = args.symbols
        if not symbols:
            try:
                kcfg = load_yaml("configs/kraken.yaml").get("kraken", {})
                symbols = kcfg.get("symbols", ["XBT/USD"]) or ["XBT/USD"]
            except Exception:
                symbols = ["XBT/USD"]

        # prepare optional retrainer (one per controller run)
        retrainer = None
        if args.enable_retrainer:
            policy = RetrainPolicy(
                metric=args.retrain_metric,
                direction=args.retrain_direction,
                threshold=args.retrain_threshold,
                window=args.retrain_window,
                cooldown_seconds=args.retrain_cooldown,
                detection_mode=args.retrain_mode,
                ewma_alpha=args.retrain_ewma_alpha,
            )

            def make_retrain_fn(candidate, out_dir):
                # retrain function invoked by the OnlineRetrainer; uses example_retrain_fn
                # when controller is not allowed to execute, otherwise triggers
                # training via train_candidate_from_csv using the provided candidate.
                def _fn(context):
                    if args.execute and args.prices_csv and candidate:
                        # perform candidate training (blocking) and return a produced artifact path
                        try:
                            train_candidate_from_csv(candidate, out_dir, args.prices_csv, args.artifacts_root)
                            # best-effort: look for created candidate_model.pkl
                            maybe = Path(out_dir) / 'candidate_model.pkl'
                            if maybe.exists():
                                return str(maybe)
                        except Exception:
                            LOG.exception('retrain training job failed')
                        # fallback to example artifact
                        return example_retrain_fn(context)
                    else:
                        return example_retrain_fn(context)

                return _fn

            retrainer = OnlineRetrainer(retrain_fn=example_retrain_fn, policy=policy, run_logger_db=args.retrain_db)
        # retrain executor (non-blocking jobs)
        retrain_executor = None
        if retrainer is not None:
            retrain_executor = ThreadPoolExecutor(max_workers=1)

        # run tuning in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max(1, int(args.parallel))) as ex:
            futures = {ex.submit(_run_tuning_for_symbol, s, args): s for s in symbols}
            for fut in as_completed(futures):
                try:
                    sym, out_dir, results_path, candidate_metrics, run_id = fut.result()
                    results.append((sym, out_dir, results_path, candidate_metrics, run_id))
                except Exception as e:
                    LOG.exception("tuning worker failed: %s", e)

        # process results sequentially (promotion, safety, training)
        for sym, out_dir, results_path, candidate_metrics, run_id in results:
            if not results_path:
                LOG.warning("no results for %s (out %s)", sym, out_dir)
                continue

            LOG.info("candidate metrics for %s = %s", sym, candidate_metrics)

            # run safety checks (placeholder state; integrate real runtime state here)
            state = {"current_exposure": 0.0, "daily_loss": 0.0}
            cfg = None
            try:
                cfg = load_yaml("configs/project.yaml").get("project", {})
            except Exception:
                cfg = {}

            safe = True
            try:
                safe = safety.check_limits(state, cfg=cfg)
            except Exception:
                LOG.exception("safety check failed for %s", sym)
                safe = False

            if not safe:
                LOG.warning("safety forbids promotion for %s; skipping", sym)
                emit_metric("promotion_blocked_safety", {"symbol": sym, "run_id": run_id})
                continue

            # Extract best candidate from tuning results and optionally train a checkpoint
            try:
                candidate = extract_best_candidate(results_path, out_dir)
                if candidate:
                    LOG.info("extracted best candidate for %s: %s", sym, candidate.get("best_params"))
                    emit_metric("candidate_extracted", {"symbol": sym, "run_id": run_id, "best_score": candidate.get("best_score")})
                    # If execution allowed, run training to create checkpoint and register it
                    if args.execute and args.prices_csv:
                        train_candidate_from_csv(candidate, out_dir, args.prices_csv, args.artifacts_root)
                # Feed retrainer with candidate metrics (if enabled)
                try:
                    if retrainer is not None:
                        # swap retrain function to capture current candidate/out_dir
                        retrainer.retrain_fn = make_retrain_fn(candidate, out_dir)
                        triggered, retrain_callable = retrainer.ingest_nonblocking(candidate_metrics or {})
                        if triggered and retrain_callable is not None:
                            emit_metric('retrain_scheduled', {'symbol': sym, 'run_id': run_id})
                            LOG.info('Scheduling retrain job for %s (run %s)', sym, run_id)
                            # submit non-blocking job
                            if retrain_executor is not None:
                                fut = retrain_executor.submit(retrain_callable)
                                # attach callback to log artifact when done
                                def _on_done(fut, symbol=sym):
                                    try:
                                        artifact = fut.result()
                                        # log artifact to RunLogger
                                        run_id2 = f"auto_retrain_{int(time.time())}"
                                        try:
                                            with RunLogger(run_id=run_id2, cfg={'policy': policy.__dict__}, db_url=args.retrain_db) as rl:
                                                if artifact:
                                                    rl.log_artifact(str(artifact), kind='model')
                                                rl.log_metrics({f'retrain_trigger_{policy.metric}': float(candidate_metrics.get(policy.metric) or 0.0)})
                                        except Exception:
                                            LOG.exception('failed to log retrain artifact')
                                        emit_metric('retrain_finished', {'symbol': sym, 'run_id': run_id, 'artifact': artifact})
                                        LOG.info('Retrain job finished for %s -> %s', symbol, artifact)
                                    except Exception:
                                        LOG.exception('retrain job failed for %s', symbol)

                                fut.add_done_callback(_on_done)
                except Exception:
                    LOG.exception('retrainer ingestion failed for %s', sym)
                else:
                    LOG.info("no candidate extracted for %s", sym)
            except Exception:
                LOG.exception("failed to extract/train candidate for %s", sym)

            try:
                from src.operations.promotion import should_promote

                promote, reason = should_promote(candidate_metrics)
            except Exception:
                promote, reason = False, "promotion check failed"

            LOG.info("promotion decision for %s: %s (%s)", sym, promote, reason)
            emit_metric("promotion_decision", {"symbol": sym, "run_id": run_id, "promote": bool(promote), "reason": reason})

            if promote and args.execute:
                train_args = [
                    "--model",
                    "rl",
                    "--steps",
                    "500",
                    "--save",
                    os.path.join(out_dir, f"model_{sym.replace('/', '-')}.pth"),
                    "--out",
                    args.artifacts_root,
                    "--register",
                ]
                if args.prices_csv:
                    # prepare replay parquet under data/raw/{PAIR}/... so RL trainer can load it
                    try:
                        data_root_for_train = "data/raw"
                        created = csv_to_replay_parquet(args.prices_csv, sym, data_root=data_root_for_train)
                        if created:
                            train_args.extend(["--replay-pair", sym, "--data-root", data_root_for_train])
                        else:
                            LOG.warning("failed to generate replay parquet for %s; invoking RL training without replay", sym)
                    except Exception:
                        LOG.exception("failed to prepare replay parquet for %s", sym)
                emit_metric("training_started", {"symbol": sym, "run_id": run_id})
                try:
                    run_module("orchestration.pipelines.training_pipeline", train_args)
                    emit_metric("training_finished", {"symbol": sym, "run_id": run_id})
                except subprocess.CalledProcessError as e:
                    LOG.exception("training failed for %s", sym)
                    emit_metric("training_failed", {"symbol": sym, "run_id": run_id, "err": str(e)})
                
                    # attempt promotion even if RL training failed earlier (ML candidate may exist)
                # After training (RL or fallback), attempt to promote the artifact directory
                try:
                    promo_args = [out_dir, "--models-dir", "models", "--promotions-file", "experiments/promotions.json"]
                    if not args.execute:
                        # do not register in DB when not executing
                        promo_args.append("--no-register")
                    emit_metric("promotion_start", {"symbol": sym, "run_id": run_id})
                    run_module("scripts.promote_candidate", promo_args)
                    emit_metric("promotion_finished", {"symbol": sym, "run_id": run_id})
                except subprocess.CalledProcessError:
                    LOG.exception("promotion script failed for %s (artifact %s)", sym, out_dir)
                    emit_metric("promotion_failed", {"symbol": sym, "run_id": run_id})

    # Run according to mode
    if args.once:
        run_once()
        return 0

    if args.daemon:
        LOG.info("Starting in daemon mode; interval=%s seconds", args.interval)
        while not stop_event.is_set():
            try:
                run_once()
            except Exception:
                LOG.exception("Exception in daemon run loop")
            # wait for interval or exit early if signal received
            stop_event.wait(args.interval)
        LOG.info("Daemon stopping")
        return 0

    # default: single run
    run_once()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
