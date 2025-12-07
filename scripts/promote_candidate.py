#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Optional

from src.persistence.db import RunLogger


def find_best_json(artifact_dir: str) -> Optional[str]:
    cand = os.path.join(artifact_dir, "BEST_CANDIDATE.json")
    if os.path.exists(cand):
        return cand
    # fallback: look for results.json and pick best fold
    res = os.path.join(artifact_dir, "results.json")
    if os.path.exists(res):
        return res
    # search deeper
    for root, dirs, files in os.walk(artifact_dir):
        if "BEST_CANDIDATE.json" in files:
            return os.path.join(root, "BEST_CANDIDATE.json")
        if "results.json" in files:
            return os.path.join(root, "results.json")
    return None


def find_candidate_checkpoint(artifact_dir: str) -> Optional[str]:
    # common candidate names
    names = ["candidate_model.pkl", "candidate_model.pth", "candidate_model.pt"]
    for n in names:
        path = os.path.join(artifact_dir, n)
        if os.path.exists(path):
            return path
    # search for pth/pkl under dir
    for root, dirs, files in os.walk(artifact_dir):
        for f in files:
            if f.endswith(".pth") or f.endswith(".pt") or f.endswith(".pkl"):
                return os.path.join(root, f)
    return None


def find_newest_checkpoint(artifact_dir: str) -> Optional[str]:
    best = None
    best_mtime = 0
    # allow file directly specified
    if os.path.isfile(artifact_dir) and (artifact_dir.endswith('.pth') or artifact_dir.endswith('.pt') or artifact_dir.endswith('.pkl')):
        return artifact_dir
    for root, dirs, files in os.walk(artifact_dir):
        for f in files:
            if f.endswith('.pth') or f.endswith('.pt') or f.endswith('.pkl'):
                p = os.path.join(root, f)
                try:
                    m = os.path.getmtime(p)
                except Exception:
                    m = 0
                if m > best_mtime:
                    best_mtime = m
                    best = p
    return best


def promote(artifact_dir: str, models_dir: str, promotions_file: str, register: bool):
    best_path = find_best_json(artifact_dir)
    if not best_path:
        print("No BEST_CANDIDATE.json or results.json found under", artifact_dir)
        # we'll synthesize minimal metadata below if possible
        best_path = None

    # load candidate info if present
    data = None
    if best_path:
        try:
            with open(best_path, "r", encoding="utf-8-sig") as fh:
                data = json.load(fh)
        except Exception as e:
            print("Warning: failed to parse candidate JSON at", best_path, "error:", e)
            data = None

    # If no candidate JSON, synthesize minimal metadata
    if data is None:
        run_id = os.path.basename(artifact_dir.rstrip(os.sep))
        data = {"run_id": run_id, "best_score": None, "best_params": {}, "metrics": {}}

    run_id = data.get("run_id") or os.path.basename(artifact_dir)
    best_params = data.get("best_params") or {}
    metrics = data.get("metrics") or {}

    # allow caller to override checkpoint by env/arg later (handled in CLI)
    ckpt = find_candidate_checkpoint(artifact_dir)
    if not ckpt:
        # fallback: find newest checkpoint anywhere under the artifact dir (or artifact_dir itself)
        ckpt = find_newest_checkpoint(artifact_dir)
    if not ckpt:
        print("No checkpoint found under", artifact_dir)
        return 3

    os.makedirs(models_dir, exist_ok=True)
    base_name = os.path.basename(ckpt)
    # create deterministic promoted name
    promoted_name = f"promoted_{run_id}_{base_name}"
    promoted_path = os.path.join(models_dir, promoted_name)
    shutil.copy2(ckpt, promoted_path)

    # Record promotion in promotions file
    promo = {
        "promoted_at": datetime.utcnow().isoformat(),
        "run_id": run_id,
        "artifact_dir": os.path.abspath(artifact_dir),
        "promoted_path": os.path.abspath(promoted_path),
        "best_params": best_params,
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(promotions_file), exist_ok=True)
    try:
        existing = []
        if os.path.exists(promotions_file):
            with open(promotions_file, "r", encoding="utf-8") as pf:
                existing = json.load(pf) or []
        existing.append(promo)
        with open(promotions_file, "w", encoding="utf-8") as pf:
            json.dump(existing, pf, indent=2)
    except Exception as e:
        print("Failed to write promotions file:", e)

    if register:
        try:
            with RunLogger(run_id, cfg={"promotion": True, "promoted_path": promoted_path}) as rl:
                rl.log_artifact(promoted_path, kind="promoted_checkpoint")
                # log metrics if any
                try:
                    rl.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
                except Exception:
                    pass
        except Exception as e:
            print("Failed to register promotion in registry DB:", e)

    print("Promoted checkpoint to:", promoted_path)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("artifact_dir", help="Path to tuning artifact directory (the controller_<id>_<PAIR> folder)")
    p.add_argument("--checkpoint", help="Optional explicit checkpoint file to promote (overrides search)")
    p.add_argument("--models-dir", default="models", help="Directory to copy promoted models to")
    p.add_argument("--promotions-file", default="experiments/promotions.json", help="Promotions index file")
    p.add_argument("--no-register", dest="register", action="store_false", help="Do not register promotion in experiments DB")
    p.set_defaults(register=True)
    args = p.parse_args(argv)
    # If explicit checkpoint provided, and artifact_dir is a folder, copy checkpoint into that folder for deterministic naming
    if args.checkpoint:
        ck = os.path.abspath(args.checkpoint)
        if not os.path.exists(ck):
            print("Explicit checkpoint file not found:", ck)
            return 4
        # if artifact_dir is a directory, copy checkpoint into it so promote can find it and log artifact path
        if os.path.isdir(args.artifact_dir):
            try:
                dest = os.path.join(args.artifact_dir, os.path.basename(ck))
                # if source and dest are identical, skip copy
                if os.path.abspath(ck) != os.path.abspath(dest):
                    shutil.copy2(ck, dest)
                args.artifact_dir = args.artifact_dir
            except Exception as e:
                print("Failed to copy checkpoint into artifact dir:", e)
                return 5
        else:
            # if artifact_dir is a file path, allow using that file as the checkpoint
            pass

    return promote(args.artifact_dir, args.models_dir, args.promotions_file, args.register)


if __name__ == "__main__":
    sys.exit(main())
