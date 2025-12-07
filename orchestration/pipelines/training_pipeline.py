from __future__ import annotations

import argparse
import os
import uuid
import logging
import sys
from typing import Optional

import pandas as pd
from src.utils.io import load_prices_csv

LOG = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/project.yaml', help='Path to project config')
    p.add_argument('--config-check', action='store_true', help='Validate config and exit')
    p.add_argument("--model", type=str, default="rl", choices=("rl", "ml", "transformer"))
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--save", type=str, default="models/ppo.pth")
    p.add_argument("--seeds", type=str, default="0", help="Comma separated seeds to run as an ensemble, e.g. '0,1,2'")
    p.add_argument("--register", action="store_true", help="Register runs/artifacts in experiments DB")
    p.add_argument("--replay-pair", type=str, default=None)
    p.add_argument("--data-root", type=str, default="data/raw")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="experiments/artifacts")
    p.add_argument("--ensemble-names", type=str, default=None, help="Comma separated model names to create ensemble manager and persist weights")
    p.add_argument("--ensemble-map", type=str, default=None, help="Comma separated mappings name=path to explicitly map ensemble names to checkpoint files")
    p.add_argument("--compute-ensemble-weights", action='store_true', help="Compute ensemble weights from available per-model predictions and val labels if present")
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--obs-mode", type=str, default="raw")
    p.add_argument("--regime-enable", action="store_true", help="Enable regime detection from prices CSV")
    p.add_argument("--regime-prices-csv", type=str, default=None, help="CSV path with timestamp and close for regime detection")
    p.add_argument("--regime-window", type=int, default=252, help="Rolling window for regime volatility detection")
    p.add_argument("--regime-sample-mode", type=str, default="per-regime", choices=("per-regime", "balanced"), help="Sampling mode when building per-regime datasets")
    p.add_argument("--regime-sample-size", type=int, default=0, help="If >0, target number of rows per regime (will sample/upsample accordingly)")
    p.add_argument("--dedupe", type=str, default="first", choices=("none", "first", "last", "mean"), help="How to dedupe duplicate timestamps when loading CSVs")
    args = p.parse_args()

    # Optional config validation hook
    if getattr(args, 'config_check', False):
        try:
            # Import the validator and run
            from tooling.validate_config import run as validate_run
            rc = validate_run([args.config])
            if rc != 0:
                print('Config validation failed')
                raise SystemExit(rc)
            print('Config validation passed')
            raise SystemExit(0)
        except SystemExit:
            raise
        except Exception as e:
            print('Config validation failed (exception):', e)
            raise SystemExit(3)

    os.makedirs(args.out, exist_ok=True)
    run_id = str(uuid.uuid4())
    out_dir = os.path.join(args.out, run_id)
    os.makedirs(out_dir, exist_ok=True)
    artifacts = []

    # optional regime detection: when enabled, try to load a prices CSV and detect regimes
    regimes_series = None
    unique_regimes = []
    if args.regime_enable:
        if args.regime_prices_csv and os.path.exists(args.regime_prices_csv):
            try:
                df_prices = load_prices_csv(args.regime_prices_csv, dedupe=args.dedupe)
                if 'timestamp' in df_prices.columns:
                    df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'], utc=True)
                    df_prices = df_prices.set_index('timestamp')
                # expect a `close` column
                if 'close' in df_prices.columns:
                    try:
                        from src.features.regime import detect_regimes
                    except Exception:
                        LOG.exception('failed to import detect_regimes')
                        detect_regimes = None
                    if detect_regimes is not None:
                        regimes_series = detect_regimes(df_prices['close'], window=args.regime_window)
                        regimes_csv = os.path.join(out_dir, 'regimes.csv')
                        os.makedirs(out_dir, exist_ok=True)
                        regimes_series.to_csv(regimes_csv, index=True, header=['regime'])
                        unique_regimes = list(sorted(set(regimes_series.dropna().unique())))
                        print(f"Detected regimes: {unique_regimes}; saved to {regimes_csv}")
                else:
                    LOG.warning("regime prices csv missing 'close' column: %s", args.regime_prices_csv)
            except Exception:
                LOG.exception('failed to detect regimes from %s', args.regime_prices_csv)
        else:
            LOG.warning('regime_enable requested but no valid --regime-prices-csv provided')

    if args.model == "rl":
        # run RL trainer as a module entry point
        try:
            from src.models.rl import train_ppo as train_ppo_cli
        except Exception as e:
            LOG.exception("failed to import rl trainer: %s", e)
            raise
        # construct argv for train_ppo.main
        argv = ["train_ppo.py", "--steps", str(args.steps), "--save", args.save, "--seed", str(args.seed), "--action-scale", str(args.action_scale), "--obs-mode", args.obs_mode]
        if args.replay_pair:
            argv.extend(["--replay-pair", args.replay_pair, "--data-root", args.data_root])
        # ensure output dir exists and run for each seed
        seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
        artifacts = []
        from src.persistence.db import RunLogger

        # If regimes were detected, train per-regime checkpoints (best-effort). Otherwise, train normally per-seed.
        if regimes_series is not None and unique_regimes:
            for regime in unique_regimes:
                safe_regime = str(regime).replace(' ', '_')
                for s in seeds:
                    argv_seed = list(argv)
                    # ensure seed override
                    if '--seed' in argv_seed:
                        idx = argv_seed.index('--seed')
                        argv_seed[idx+1] = str(s)
                    else:
                        argv_seed.extend(['--seed', str(s)])
                    # ensure per-regime save path
                    save_path = args.save
                    base, ext = os.path.splitext(save_path)
                    save_path_reg = f"{base}_{safe_regime}_seed{s}{ext}"
                    if '--save' in argv_seed:
                        i = argv_seed.index('--save')
                        argv_seed[i+1] = save_path_reg
                    else:
                        argv_seed.extend(['--save', save_path_reg])
                    # if we have a replay_pair configured, create a per-regime CSV (per-regime-only) and pass via --replay-csv
                    replay_csv = None
                    try:
                        if args.replay_pair:
                            try:
                                from src.models.rl.data import load_price_history
                            except Exception:
                                load_price_history = None
                            if load_price_history is not None:
                                prices_full = load_price_history(args.data_root or 'data/raw', args.replay_pair)
                                if not prices_full.empty and regimes_series is not None:
                                    pf = prices_full.copy()
                                    pf = pf.sort_values('timestamp')
                                    regimes_df = regimes_series.reset_index()
                                    regimes_df.columns = ['timestamp', 'regime']
                                    regimes_df = regimes_df.sort_values('timestamp')
                                    merged = pd.merge_asof(pf, regimes_df, on='timestamp', direction='backward')
                                    merged = merged.dropna(subset=['regime'])
                                    # select only rows for this regime
                                    sel = merged[merged['regime'] == regime].copy()
                                    if sel.empty:
                                        LOG.warning('no price rows found for regime %s; falling back to full price history', regime)
                                        sel = merged.copy()
                                    # apply sampling/upsample if requested
                                    target = int(args.regime_sample_size) if int(args.regime_sample_size) > 0 else None
                                    if target is not None:
                                        if len(sel) >= target:
                                            sel = sel.sample(n=target, random_state=int(s))
                                        else:
                                            sel = sel.sample(n=target, replace=True, random_state=int(s))
                                    # if sample_mode is 'balanced' and target is None, try to sample across regimes (fallback to per-regime)
                                    sample_dir = os.path.join(out_dir, 'regime_samples')
                                    os.makedirs(sample_dir, exist_ok=True)
                                    replay_csv = os.path.join(sample_dir, f"prices_{safe_regime}_seed{s}.csv")
                                    sel.to_csv(replay_csv, index=False)
                                    argv_seed.extend(['--replay-csv', replay_csv])
                    except Exception:
                        LOG.exception('failed to build per-regime replay CSV for regime training')

                    prev = list(sys.argv)
                    try:
                        sys.argv = argv_seed
                        train_ppo_cli.main()
                    finally:
                        sys.argv = prev
                    artifacts.append(save_path_reg)
                    if args.register:
                        run_id = str(uuid.uuid4())
                        cfg = {"model": args.model, "save_path": save_path_reg, "seed": s, "steps": args.steps, "regime": safe_regime}
                        with RunLogger(run_id, cfg=cfg) as rl:
                            rl.log_artifact(save_path_reg, kind='checkpoint')
        else:
            for s in seeds:
                argv_seed = list(argv)
                # ensure seed override
                if '--seed' in argv_seed:
                    idx = argv_seed.index('--seed')
                    argv_seed[idx+1] = str(s)
                else:
                    argv_seed.extend(['--seed', str(s)])
                # ensure per-seed save path
                save_path = args.save
                if seeds and len(seeds) > 1:
                    base, ext = os.path.splitext(save_path)
                    save_path = f"{base}_seed{s}{ext}"
                if '--save' in argv_seed:
                    i = argv_seed.index('--save')
                    argv_seed[i+1] = save_path
                else:
                    argv_seed.extend(['--save', save_path])
                prev = list(sys.argv)
                try:
                    sys.argv = argv_seed
                    train_ppo_cli.main()
                finally:
                    sys.argv = prev
                artifacts.append(save_path)
                if args.register:
                    run_id = str(uuid.uuid4())
                    cfg = {"model": args.model, "save_path": save_path, "seed": s, "steps": args.steps}
                    with RunLogger(run_id, cfg=cfg) as rl:
                        rl.log_artifact(save_path, kind='checkpoint')
        # print artifacts
        print(f"Training completed; artifacts: {artifacts}")

    elif args.model == "ml":
        # Simple hook to run the minimal ML trainer (placeholder implementation)
        try:
            from src.training.trainer import train_ml
        except Exception as e:
            LOG.exception("failed to import ml trainer: %s", e)
            raise
        # determine save path
        save_path = args.save
        artifacts = []
        # If regimes detected, train one ML model per regime (best-effort) and save per-regime checkpoints
        if regimes_series is not None and unique_regimes:
            # Build per-regime datasets (per-regime-only) using CSVs under data_root (best-effort).
            import glob
            try:
                csv_paths = glob.glob(os.path.join(args.data_root or 'data/raw', '*.csv')) if args.data_root else []
                parts = []
                for p in csv_paths:
                    try:
                        dft = load_prices_csv(p, dedupe=args.dedupe)
                        parts.append(dft)
                    except Exception:
                        continue
                df_all = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            except Exception:
                LOG.exception('failed to read CSVs for per-regime ML training')
                df_all = pd.DataFrame()

            regimes_df = regimes_series.reset_index() if regimes_series is not None else pd.DataFrame()
            if not regimes_df.empty:
                regimes_df.columns = ['timestamp', 'regime']
                regimes_df = regimes_df.sort_values('timestamp')

            for regime in unique_regimes:
                safe_regime = str(regime).replace(' ', '_')
                base, ext = os.path.splitext(save_path)
                save_path_reg = f"{base}_{safe_regime}{ext}"
                try:
                    # prepare per-regime DataFrame
                    sample_df = None
                    if not df_all.empty and not regimes_df.empty and 'timestamp' in df_all.columns:
                        try:
                            df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], utc=True)
                            df_all = df_all.sort_values('timestamp')
                            merged = pd.merge_asof(df_all, regimes_df, on='timestamp', direction='backward')
                            merged = merged.dropna(subset=['regime'])
                            sel = merged[merged['regime'] == regime].copy()
                            if sel.empty:
                                LOG.warning('no CSV rows found for regime %s; falling back to merged', regime)
                                sel = merged.copy()
                            target = int(args.regime_sample_size) if int(args.regime_sample_size) > 0 else None
                            if target is not None:
                                if len(sel) >= target:
                                    sel = sel.sample(n=target, random_state=int(args.seed))
                                else:
                                    sel = sel.sample(n=target, replace=True, random_state=int(args.seed))
                            sample_df = sel.reset_index(drop=True)
                        except Exception:
                            LOG.exception('failed to build per-regime DataFrame for regime %s', regime)
                            sample_df = None

                    if sample_df is not None:
                        saved = train_ml(data_root=args.data_root, save=save_path_reg, steps=args.steps, seed=args.seed, data_df=sample_df)
                    else:
                        saved = train_ml(data_root=args.data_root, save=save_path_reg, steps=args.steps, seed=args.seed)
                except Exception:
                    LOG.exception('failed to run train_ml for regime %s', safe_regime)
                    saved = save_path_reg
                artifacts.append(saved)
                if args.register:
                    run_id = str(uuid.uuid4())
                    cfg = {"model": args.model, "save_path": saved, "seed": args.seed, "steps": args.steps, "regime": safe_regime}
                    from src.persistence.db import RunLogger
                    # register checkpoint and metrics if present
                    metrics_path = None
                    try:
                        base2, _ = os.path.splitext(saved)
                        metrics_path = f"{base2}.metrics.json"
                    except Exception:
                        metrics_path = None

                    with RunLogger(run_id, cfg=cfg) as rl:
                        rl.log_artifact(saved, kind='checkpoint')
                        if metrics_path and os.path.exists(metrics_path):
                            try:
                                import json

                                with open(metrics_path, 'r', encoding='utf-8') as mf:
                                    metrics = json.load(mf)
                                if isinstance(metrics, dict):
                                    rl.log_metrics({k: float(v) for k, v in metrics.items()})
                            except Exception:
                                pass
                            rl.log_artifact(metrics_path, kind='metrics')
        else:
            # run trainer normally
            saved = train_ml(data_root=args.data_root, save=save_path, steps=args.steps, seed=args.seed)
            artifacts = [saved]
            if args.register:
                run_id = str(uuid.uuid4())
                cfg = {"model": args.model, "save_path": saved, "seed": args.seed, "steps": args.steps}
                from src.persistence.db import RunLogger
                metrics_path = None
                try:
                    base, _ = os.path.splitext(saved)
                    metrics_path = f"{base}.metrics.json"
                except Exception:
                    metrics_path = None

                with RunLogger(run_id, cfg=cfg) as rl:
                    rl.log_artifact(saved, kind='checkpoint')
                    if metrics_path and os.path.exists(metrics_path):
                        try:
                            import json

                            with open(metrics_path, 'r', encoding='utf-8') as mf:
                                metrics = json.load(mf)
                            if isinstance(metrics, dict):
                                rl.log_metrics({k: float(v) for k, v in metrics.items()})
                        except Exception:
                            pass
                        rl.log_artifact(metrics_path, kind='metrics')
        print(f"ML training completed; artifacts: {artifacts}")

    # If an ensemble spec was provided, create a default EnsembleManager and persist initial weights
    if args.ensemble_names:
        try:
            names = [n.strip() for n in str(args.ensemble_names).split(',') if n.strip()]
            if names:
                try:
                    from src.models.ensemble_manager import EnsembleManager
                except Exception:
                    LOG.exception('failed to import EnsembleManager')
                    raise
                import json
                weights_path = os.path.join(out_dir, 'ensemble_weights.json')
                em = EnsembleManager(names)
                try:
                    with open(weights_path, 'w', encoding='utf-8') as wf:
                        # build initial checkpoint mapping
                        checkpoints = {n: None for n in em.model_names}

                        # 1) parse explicit mappings if provided (name=path,...)
                        explicit = {}
                        if args.ensemble_map:
                            for pair in str(args.ensemble_map).split(','):
                                if '=' in pair:
                                    k, v = pair.split('=', 1)
                                    k = k.strip()
                                    v = v.strip()
                                    if k:
                                        explicit[k] = v

                        # helper to copy mapped files into run dir for reproducibility
                        import shutil
                        ckpt_out_dir = os.path.join(out_dir, 'checkpoints')
                        os.makedirs(ckpt_out_dir, exist_ok=True)

                        # apply explicit mappings first
                        for name in em.model_names:
                            if name in explicit:
                                src = explicit[name]
                                try:
                                    if os.path.exists(src):
                                        dst = os.path.join(ckpt_out_dir, os.path.basename(src))
                                        shutil.copy2(src, dst)
                                        checkpoints[name] = os.path.abspath(dst)
                                        # copy sibling preds/labels files if present
                                        base_src = os.path.splitext(src)[0]
                                        for suf in ('.preds.npy', '.val_labels.npy'):
                                            sibling = f"{base_src}{suf}"
                                            try:
                                                if os.path.exists(sibling):
                                                    shutil.copy2(sibling, os.path.join(ckpt_out_dir, os.path.basename(sibling)))
                                            except Exception:
                                                continue
                                    else:
                                        LOG.warning('explicit ensemble-map path for %s does not exist: %s', name, src)
                                except Exception:
                                    LOG.exception('failed to copy explicit checkpoint for %s', name)

                        # 2) match remaining names against produced artifacts by basename contains name
                        remaining_names = [n for n, v in checkpoints.items() if v is None]
                        if remaining_names:
                            for name in remaining_names:
                                matched = None
                                for a in artifacts:
                                    try:
                                        if name in os.path.basename(a):
                                            try:
                                                dst = os.path.join(ckpt_out_dir, os.path.basename(a))
                                                shutil.copy2(a, dst)
                                                matched = os.path.abspath(dst)
                                                # also copy sibling preds/labels
                                                base_a = os.path.splitext(a)[0]
                                                for suf in ('.preds.npy', '.val_labels.npy'):
                                                    sib = f"{base_a}{suf}"
                                                    try:
                                                        if os.path.exists(sib):
                                                            shutil.copy2(sib, os.path.join(ckpt_out_dir, os.path.basename(sib)))
                                                    except Exception:
                                                        continue
                                                break
                                            except Exception:
                                                matched = os.path.abspath(a)
                                                break
                                    except Exception:
                                        continue
                                checkpoints[name] = matched

                        # 3) final fallback: if still none and counts align, map by index
                        if all(v is None for v in checkpoints.values()) and len(artifacts) == len(em.model_names):
                            for i, name in enumerate(em.model_names):
                                try:
                                    a = artifacts[i]
                                    dst = os.path.join(ckpt_out_dir, os.path.basename(a))
                                    try:
                                        shutil.copy2(a, dst)
                                        checkpoints[name] = os.path.abspath(dst)
                                    except Exception:
                                        checkpoints[name] = os.path.abspath(a)
                                except Exception:
                                    checkpoints[name] = None

                        # optionally include a simple regime->checkpoint mapping
                        regime_map = {}
                        if unique_regimes:
                            try:
                                # if we have at least as many artifacts as regimes, map by index
                                if len(artifacts) >= len(unique_regimes):
                                    for i, r in enumerate(unique_regimes):
                                        a = None
                                        try:
                                            if i < len(artifacts):
                                                src_a = artifacts[i]
                                                if os.path.exists(src_a):
                                                    dst = os.path.join(ckpt_out_dir, os.path.basename(src_a))
                                                    shutil.copy2(src_a, dst)
                                                    a = os.path.abspath(dst)
                                                else:
                                                    a = os.path.abspath(src_a)
                                        except Exception:
                                            a = None
                                        regime_map[r] = a
                                else:
                                    # fallback: assign the first available checkpoint to all regimes
                                    first_ckpt = next((v for v in checkpoints.values() if v), None)
                                    for r in unique_regimes:
                                        regime_map[r] = first_ckpt
                            except Exception:
                                LOG.exception('failed to build regime_map')

                        json.dump({'model_names': em.model_names, 'weights': em.get_weights(), 'checkpoints': checkpoints, 'regime_map': regime_map}, wf, indent=2)
                        # attempt to compute ensemble weights if requested and validation preds exist
                        if args.compute_ensemble_weights:
                            try:
                                import glob as _glob
                                import numpy as _np

                                preds = {}
                                val_y = None
                                # load any preds files under ckpt_out_dir
                                for p in _glob.glob(os.path.join(ckpt_out_dir, '*.preds.npy')):
                                    name = os.path.basename(p)
                                    # try to infer model name from filename (strip extensions)
                                    key = name.replace('.preds.npy', '')
                                    try:
                                        preds[key] = _np.load(p)
                                    except Exception:
                                        continue
                                # prefer a val_labels.npy if present
                                val_paths = _glob.glob(os.path.join(ckpt_out_dir, '*val_labels.npy'))
                                if val_paths:
                                    try:
                                        val_y = _np.load(val_paths[0])
                                    except Exception:
                                        val_y = None

                                # map preds keys to ensemble model names heuristically
                                mapped_preds = {}
                                for m in em.model_names:
                                    # exact match
                                    if m in preds:
                                        mapped_preds[m] = preds[m]
                                        continue
                                    # substring match
                                    for k in preds.keys():
                                        if m in k or k in m:
                                            mapped_preds[m] = preds[k]
                                            break
                                if mapped_preds and val_y is not None:
                                    try:
                                        new_weights = em.fit_weights_from_preds(mapped_preds, val_y, metric='mse')
                                        # write back updated weights
                                        wf.seek(0)
                                        json.dump({'model_names': em.model_names, 'weights': em.get_weights(), 'checkpoints': checkpoints, 'regime_map': regime_map}, wf, indent=2)
                                        wf.truncate()
                                        print(f"Computed ensemble weights from validation preds: {new_weights}")
                                    except Exception:
                                        LOG.exception('failed to fit ensemble weights from preds')
                            except Exception:
                                LOG.exception('error while attempting to compute ensemble weights')
                    print(f"Saved ensemble weights to {weights_path}")
                    artifacts.append(weights_path)
                    if args.register:
                        from src.persistence.db import RunLogger
                        run_id2 = str(uuid.uuid4())
                        cfg = {"ensemble": names, "weights_path": weights_path}
                        with RunLogger(run_id2, cfg=cfg) as rl:
                            rl.log_artifact(weights_path, kind='ensemble')
                except Exception:
                    LOG.exception('failed to write ensemble weights file')
        except Exception:
            LOG.exception('failed to setup ensemble manager')

    print(f"Training completed; artifacts in {out_dir}")


if __name__ == "__main__":
    main()
