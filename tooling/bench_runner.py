"""Simple benchmarking harness to run commands in parallel and collect logs.

Features added:
- per-job start/end timestamps
- optional per-job timeout and retry support
- summary CSV with metadata (cmd, log, returncode, start, end, attempts, duration, reason)

Usage example:
  python tooling/bench_runner.py --commands-file experiments.txt --parallel 3 --out-dir results/bench --timeout 300 --retries 1

This is intentionally small and dependency-free. Commands are executed with
shell=True to preserve platform shell behavior (PowerShell on Windows).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.time import now_iso
from pathlib import Path
from typing import List, Dict, Optional


def _now_iso() -> str:
    return now_iso() + "Z"


def run_job(cmd: str, logpath: str, timeout: Optional[int], retries: int) -> Dict:
    """Run a single command, writing stdout/stderr to logpath. Returns metadata dict.

    Retries on non-zero exit up to `retries` times. timeout is per-attempt seconds.
    """
    attempts = 0
    start_ts = _now_iso()
    start_time = time.time()
    reason = "ok"
    returncode = None

    attempt_results = []
    while attempts <= retries:
        attempts += 1
        with open(logpath, "ab") as f:
            f.write(f"\n--- ATTEMPT {attempts} START { _now_iso() } ---\n".encode())
            f.flush()
            try:
                # subprocess.run simplifies timeout handling
                completed = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
                returncode = completed.returncode
                if returncode == 0:
                    reason = "ok"
                    attempt_results.append({"attempt": attempts, "returncode": returncode, "reason": reason, "time": _now_iso()})
                    break
                else:
                    reason = f"exit:{returncode}"
                    attempt_results.append({"attempt": attempts, "returncode": returncode, "reason": reason, "time": _now_iso()})
            except subprocess.TimeoutExpired:
                f.write(f"\nTIMEOUT after {timeout}s\n".encode())
                reason = "timeout"
                returncode = -1
                attempt_results.append({"attempt": attempts, "returncode": returncode, "reason": reason, "time": _now_iso()})
            except Exception as exc:  # pragma: no cover - defensive
                f.write(f"\nEXCEPTION: {exc}\n".encode())
                reason = f"exception:{type(exc).__name__}"
                returncode = -2
                attempt_results.append({"attempt": attempts, "returncode": returncode, "reason": reason, "time": _now_iso()})

        # if we get here and didn't succeed, either retry (loop) or finish
        if attempts <= retries and returncode != 0:
            time.sleep(1)

    end_ts = _now_iso()
    duration = time.time() - start_time
    # write a JSON details file next to the log containing attempt history
    try:
        import json
        details_path = logpath + ".json"
        with open(details_path, "w", encoding="utf-8") as df:
            json.dump({"cmd": cmd, "start": start_ts, "end": end_ts, "attempts": attempts, "duration": duration, "attempts_detail": attempt_results}, df, indent=2)
    except Exception:
        details_path = ""

    return {
        "cmd": cmd,
        "log": logpath,
        "details": details_path,
        "returncode": returncode,
        "start": start_ts,
        "end": end_ts,
        "attempts": attempts,
        "duration": f"{duration:.2f}",
        "reason": reason,
    }


def run_commands(commands: List[str], out_dir: str, parallel: int = 2, timeout: Optional[int] = None, retries: int = 0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Use ThreadPoolExecutor to allow blocking subprocess.run with timeout while keeping easy parallelism
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {}
        for idx, cmd in enumerate(commands):
            logpath = str(logs_dir / f"cmd_{idx}.log")
            # create/clear log file
            open(logpath, "wb").close()
            fut = ex.submit(run_job, cmd, logpath, timeout, retries)
            futures[fut] = cmd

        for fut in as_completed(futures):
            meta = fut.result()
            results.append(meta)

    # write summary
    summary_path = out_dir / "summary.csv"
    # include 'details' (path to per-job JSON) so consumers can inspect attempt history
    fieldnames = ["cmd", "log", "details", "returncode", "start", "end", "attempts", "duration", "reason"]
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Write a richer JSON summary alongside the CSV (includes a small env snapshot)
    def _capture_env() -> Dict[str, str]:
        keys = ["PYTHONPATH", "PATH", "VIRTUAL_ENV", "USER", "USERNAME", "COMSPEC", "SHELL", "OS"]
        env = {}
        for k in keys:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        return env

    summary_json = out_dir / "summary.json"
    try:
        with open(summary_json, "w", encoding="utf-8") as jf:
            json.dump({"generated": _now_iso(), "env": _capture_env(), "results": results}, jf, indent=2)
    except Exception:
        # non-fatal: keep CSV as primary artifact
        pass

    print(f"finished all commands; summary: {summary_path} (json: {summary_json})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--commands-file", required=True)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--out-dir", default="results/bench")
    p.add_argument("--timeout", type=int, default=None, help="per-job timeout in seconds")
    p.add_argument("--retries", type=int, default=0, help="number of retries for failing jobs")
    args = p.parse_args()

    with open(args.commands_file, "r", encoding="utf-8") as fh:
        cmds = [line.strip() for line in fh if line.strip() and not line.strip().startswith("#")]

    run_commands(cmds, args.out_dir, parallel=args.parallel, timeout=args.timeout, retries=args.retries)


if __name__ == "__main__":
    main()
