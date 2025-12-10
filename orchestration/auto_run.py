from __future__ import annotations

import argparse
import json
import sys
import subprocess
import platform
import glob
import math
import numpy as np
import os
import shutil
import uuid
from datetime import datetime, timezone

import pandas as pd

from src.training.trainer import train_ml
from src.training.adapter import inference_backtest
from tooling.structured_logging import setup_structured_logger


def validate_prices_df(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise ValueError("prices DataFrame is empty")
    if "close" not in df.columns:
        raise ValueError("prices DataFrame must contain 'close' column")
    if df["close"].isnull().any():
        raise ValueError("prices close column contains NaNs")
    if len(df) < 20:
        raise ValueError("prices DataFrame too short (<20 rows)")


def compute_max_drawdown(pnl_series: pd.Series) -> float:
    # cumulative PnL expected; if not, take cumulative sum
    s = pnl_series.cumsum() if not pnl_series.index.is_monotonic_increasing or (pnl_series.diff().abs().sum() == 0) else pnl_series.cumsum()
    peak = s.cummax()
    dd = peak - s
    return float(dd.max()) if not dd.empty else 0.0


def _get_git_sha():
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return p.stdout.strip()
    except Exception:
        return None


def _env_info():
    info = {"python": sys.version, "platform": platform.platform()}
    # try to capture versions of key packages
    pkgs = {}
    for name in ("pandas", "numpy", "sklearn", "pyarrow"):
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            pkgs[name] = None
    info["packages"] = pkgs
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices-csv", type=str, default=None, help="Optional historical prices CSV to use for eval")
    p.add_argument("--data-root", type=str, default="data/raw")
    p.add_argument("--out-root", type=str, default="experiments/artifacts")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--auto-deploy", action="store_true", help="If set, copy model to deploy/paper on successful gate")
    p.add_argument("--max-drawdown", type=float, default=1000.0, help="Max absolute drawdown allowed for acceptance")
    p.add_argument("--min-executions", type=int, default=1, help="Minimum number of executions expected in backtest")
    p.add_argument("--min-sharpe", type=float, default=None, help="Minimum Sharpe-like ratio required to pass gate (optional)")
    p.add_argument("--max-drawdown-percentile", type=float, default=None, help="Require current drawdown to be <= percentile of past run drawdowns (0-100)")
    p.add_argument("--min-expected-return", type=float, default=None, help="Minimum average PnL change expected per step (optional)")
    p.add_argument("--pair", type=str, default="XBT/USD", help="Pair for ExchangeSimulator")
    p.add_argument("--paper-mode", action="store_true", help="Run in paper-mode: use PaperOrderExecutor (no live orders)")
    p.add_argument("--sim-seed", type=int, default=None, help="Seed for deterministic stochastic fills in simulator")
    p.add_argument("--partial-fill-ratio", type=float, default=0.9, help="Mean partial fill ratio (0-1)")
    p.add_argument("--partial-fill-std", type=float, default=0.05, help="Stddev for stochastic partial fill model")
    p.add_argument("--slippage-pct", type=float, default=0.0005, help="Fractional slippage to apply to fills (e.g. 0.0005)")
    p.add_argument("--fill-model", type=str, default="deterministic", choices=("deterministic", "stochastic"), help="Fill model for ExchangeSimulator")
    p.add_argument("--policy", type=str, default="rule", choices=("rule", "ai"), help="Decision policy to use for auto-deploy (rule or ai)")
    args = p.parse_args()

    run_id = str(uuid.uuid4())
    out_dir = os.path.join(args.out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # set up structured per-run logger
    try:
        run_logger = setup_structured_logger(f"{__name__}.{run_id}", file_path=os.path.join(out_dir, 'run.log'))
    except Exception:
        run_logger = setup_structured_logger(__name__)

    # 1) Load prices for validation/eval if provided
    prices = None
    if args.prices_csv:
        if not os.path.exists(args.prices_csv):
            raise FileNotFoundError(args.prices_cvs)
        from src.utils.io import load_prices_csv
        df = load_prices_csv(args.prices_csv, dedupe='first')
        # prefer timestamp col if present
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        prices = pd.DataFrame({"close": df["close"].astype(float)}, index=df.index)
        prices.index.name = None
        validate_prices_df(prices)

    # 2) Train model and save checkpoint into run folder
    ckpt_path = os.path.join(out_dir, "model.pkl")
    print("Training ML model...")
    train_ml(data_root=args.data_root, save=ckpt_path, steps=args.steps, seed=args.seed)

    # copy metadata & metrics next to run dir if produced by trainer
    meta_path = os.path.splitext(ckpt_path)[0] + ".meta.json"
    metrics_path = os.path.splitext(ckpt_path)[0] + ".metrics.json"
    # if metrics exist, copy to out_dir (trainer already wrote there because save was in out_dir)

    # 3) Run inference/backtest
    print("Running inference/backtest...")
    from src.execution.simulators import ExchangeSimulator
    from orchestration.decision import simple_rule_decision, ai_policy_decision
    from orchestration.alerts import notify

    # create a simulator with conservative slippage and partial-fill behavior for paper trading
    sim = ExchangeSimulator(
        pair=args.pair,
        allow_partial=True,
        partial_fill_ratio=float(args.partial_fill_ratio),
        partial_fill_std=float(args.partial_fill_std),
        slippage_pct=float(args.slippage_pct),
        fill_model=str(args.fill_model),
        seed=args.sim_seed,
    )
    # If paper-mode is requested, prefer using the PaperOrderExecutor
    from src.execution.order_executor import PaperOrderExecutor, LiveOrderExecutor
    executor = None
    if args.paper_mode:
        executor = PaperOrderExecutor(simulator=sim)
    else:
        # default: use simulator directly via an executor wrapper (dry-run LiveOrderExecutor)
        executor = None
    # if prices not provided, the inference adapter will synthesize a small series when called with None; use small default
    result = None
    try:
        # attempt to load prices from trainer data if not provided: use a small synthetic series
        if prices is None:
            idx = pd.date_range("2020-01-01", periods=120, freq="T")
            import numpy as np

            rng = np.random.default_rng(0)
            returns = rng.normal(scale=0.001, size=len(idx))
            price = 100.0 + np.cumsum(returns)
            prices = pd.DataFrame({"close": price}, index=idx)
            prices.index.name = None

        # call inference_backtest with executor if available (paper mode)
        result = inference_backtest(ckpt_path, prices, sim if executor is None else None, executor=executor, method="vol_norm", scale=1.0, vol_window=20, max_size=2.0)
        # persist a JSON-serializable result for later inspection / plotting
        try:
            serial = {}
            # pnl: Series-like -> list of [iso_ts, value]
            pnl = result.get("pnl") if isinstance(result, dict) else None
            if pnl is not None:
                try:
                    serial["pnl"] = [[str(idx), float(v)] for idx, v in zip(pnl.index.astype(str).to_list(), pnl.tolist())]
                except Exception:
                    # fallback if pnl is plain list
                    try:
                        serial["pnl"] = [[str(i), float(v)] for i, v in enumerate(pnl)]
                    except Exception:
                        serial["pnl"] = []
            # executions: try to serialize list of dicts
            executions = result.get("executions") if isinstance(result, dict) else None
            if executions is not None:
                try:
                    serial["executions"] = [dict(e) for e in executions]
                except Exception:
                    serial["executions"] = []
            if serial:
                # if executor produced fills, attach them for inspection
                try:
                    if executor is not None and hasattr(executor, 'fills'):
                        serial.setdefault('executions', [])
                        # extend with executor-recorded fills (they are dict-like)
                        serial['executions'] = list(serial.get('executions', [])) + list(executor.fills)
                except Exception:
                    pass
                with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as rf:
                    json.dump(serial, rf, indent=2)
        except Exception:
            pass
    except Exception as e:
        print("inference/backtest failed:", e)
        summary = {"run_id": run_id, "status": "failed", "reason": str(e)}
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        raise

    # 4) Compute gating metrics
    pnl = result.get("pnl") if isinstance(result, dict) else None
    executions = result.get("executions") if isinstance(result, dict) else None
    if pnl is None:
        # attempt to read as Series-like
        pnl = pd.Series([])
    max_dd = compute_max_drawdown(pnl)
    n_exec = len(executions) if executions is not None else 0

    # compute a simple Sharpe-like ratio from PnL changes
    try:
        returns = np.asarray(pnl.diff().fillna(0).values)
        if returns.size > 1:
            mean = float(np.mean(returns))
            std = float(np.std(returns, ddof=0))
            sharpe = float((mean / (std + 1e-12)) * math.sqrt(max(1, returns.size)))
        else:
            sharpe = None
    except Exception:
        sharpe = None

    # baseline comparison: search existing artifacts for best (lowest) mse if available
    baseline_mse = None
    try:
        meta_files = glob.glob(os.path.join(args.out_root, "*", "*.meta.json"))
        best = None
        for m in meta_files:
            try:
                d = json.load(open(m, "r", encoding="utf-8"))
                # metrics may be nested
                metrics = d.get("metrics") or d.get("metrics", {})
                cur = None
                if isinstance(metrics, dict):
                    cur = metrics.get("val_mse") or metrics.get("mse")
                if cur is not None:
                    cur = float(cur)
                    if best is None or cur < best:
                        best = cur
            except Exception:
                continue
        baseline_mse = best
    except Exception:
        baseline_mse = None

    # historical drawdowns for percentile-based gating
    hist_maxdds = []
    try:
        summary_files = glob.glob(os.path.join(args.out_root, "*", "summary.json"))
        for s in summary_files:
            try:
                d = json.load(open(s, "r", encoding="utf-8"))
                if isinstance(d, dict) and d.get("max_drawdown") is not None:
                    try:
                        val = d.get("max_drawdown")
                        if val is None:
                            continue
                        hist_maxdds.append(float(val))
                    except Exception:
                        continue
            except Exception:
                continue
    except Exception:
        hist_maxdds = []

    # try to read current model metrics
    current_mse = None
    try:
        if os.path.exists(meta_path):
            md = json.load(open(meta_path, "r", encoding="utf-8"))
            metrics = md.get("metrics") or {}
            tmp = metrics.get("mse") if metrics.get("mse") is not None else metrics.get("val_mse")
            current_mse = float(tmp) if tmp is not None else None
    except Exception:
        current_mse = None

    improved_vs_baseline = None
    if baseline_mse is not None and current_mse is not None:
        improved_vs_baseline = bool(current_mse < baseline_mse)

    passed = (max_dd <= float(args.max_drawdown)) and (n_exec >= int(args.min_executions))
    # optionally enforce Sharpe threshold
    if args.min_sharpe is not None:
        try:
            if sharpe is None or float(sharpe) < float(args.min_sharpe):
                passed = False
        except Exception:
            passed = False

    # optionally enforce min expected return (average step PnL change)
    if args.min_expected_return is not None:
        try:
            arr = np.asarray(pnl.diff().fillna(0).values)
            avg_ret = float(np.mean(arr)) if arr.size > 0 else 0.0
            if avg_ret < float(args.min_expected_return):
                passed = False
        except Exception:
            passed = False

    # optionally enforce drawdown percentile vs historical runs
    if args.max_drawdown_percentile is not None and hist_maxdds:
        try:
            perc = float(args.max_drawdown_percentile)
            perc_val = float(np.percentile(hist_maxdds, perc))
            # require current max_dd to be <= percentile value
            if max_dd > perc_val:
                passed = False
        except Exception:
            pass

    # additionally require improvement vs baseline if baseline exists
    if baseline_mse is not None and current_mse is not None:
        passed = passed and improved_vs_baseline

    # compute an average step return metric for decision logic
    avg_step = None
    try:
        arr = pnl.diff().fillna(0).values
        # arr may be numpy array or pandas ExtensionArray; coerce via numpy
        import numpy as _np

        _arr = _np.asarray(arr)
        avg_step = float(_arr.mean()) if _arr.size > 0 else 0.0
    except Exception:
        avg_step = None

    from src.utils.time import now_iso, now_utc

    summary = {
        "run_id": run_id,
        "created_at": now_iso(),
        "max_drawdown": max_dd,
        "n_executions": n_exec,
        "sharpe": sharpe,
        "current_mse": current_mse,
        "baseline_mse": baseline_mse,
        "avg_step_return": avg_step,
        "improved_vs_baseline": improved_vs_baseline,
        "passed": passed,
    }
    # write audit/run metadata
    try:
        run_meta = {"git_sha": _get_git_sha(), "env": _env_info(), "args": vars(args), "run_id": run_id}
        with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as mf:
            json.dump(run_meta, mf, indent=2)
    except Exception:
        pass

    # write repro.json for deterministic replay
    try:
        repro = {"run_id": run_id, "created_at": now_iso(), "args": vars(args), "sim_seed": args.sim_seed}
        with open(os.path.join(out_dir, "repro.json"), "w", encoding="utf-8") as rf:
            json.dump(repro, rf, indent=2)
    except Exception:
        pass

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("Run summary:", summary)

    # 5) Decision & Auto-deploy to paper-mode
    decision = None
    if args.policy == "ai":
        decision = ai_policy_decision(summary)
    else:
        # allow passing thresholds via CLI if needed; keep conservative defaults
        thresholds = {"max_drawdown": float(args.max_drawdown)}
        if args.min_sharpe is not None:
            thresholds["min_sharpe"] = float(args.min_sharpe)
        decision = simple_rule_decision(summary, thresholds=thresholds)

    print("Decision:", decision)
    # notify stakeholders of run summary and decision
    try:
        notify(summary, webhook_url=os.environ.get("RUN_WEBHOOK_URL"), fallback_path=os.path.join(out_dir, "alerts.log"))
    except Exception:
        pass

    if decision.get("approved") and args.auto_deploy and passed:
        deploy_dir = os.path.join("deploy", "paper", now_utc().strftime("%Y%m%dT%H%M%S"))
        os.makedirs(deploy_dir, exist_ok=True)
        shutil.copy(ckpt_path, os.path.join(deploy_dir, "model.pkl"))
        with open(os.path.join(deploy_dir, "deployed.meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"run_id": run_id, "deployed_at": now_iso(), "source": os.path.abspath(ckpt_path), "decision": decision}, fh)
        print("Deployed model to", deploy_dir)
    else:
        print("Not deploying — decision approved:", decision.get("approved"), "passed:", passed)


if __name__ == "__main__":
    main()
