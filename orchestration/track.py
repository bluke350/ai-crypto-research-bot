from __future__ import annotations

import argparse
import csv
import json
import os
from typing import List, Optional


def _collect_runs(artifacts_root: str) -> List[dict]:
    out: List[dict] = []
    if not os.path.exists(artifacts_root):
        return out
    for d in sorted(os.listdir(artifacts_root), reverse=True):
        rd = os.path.join(artifacts_root, d)
        if not os.path.isdir(rd):
            continue
        sfn = os.path.join(rd, "summary.json")
        if not os.path.exists(sfn):
            continue
        try:
            with open(sfn, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        data.setdefault("run_dir", rd)
        data.setdefault("run_id", d)
        out.append(data)
    return out


def export_runs(runs: List[dict], out_path: str, fmt: str = "csv") -> None:
    fmt = (fmt or "csv").lower()
    if fmt not in ("csv", "tsv", "json"):
        raise ValueError("unsupported export format")
    if fmt == "json":
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(runs, fh, indent=2)
        return

    delim = "," if fmt == "csv" else "\t"
    # collect keys union
    keys = set()
    for r in runs:
        keys.update(r.keys())
    keys = sorted(keys)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=delim)
        writer.writerow(keys)
        for r in runs:
            row = [r.get(k, "") for k in keys]
            writer.writerow(row)


def compare_two_runs(a: dict, b: dict) -> List[tuple]:
    # produce a compact comparison table (key, a_val, b_val)
    keys = sorted(set(list(a.keys()) + list(b.keys())))
    out: List[tuple] = []
    for k in keys:
        out.append((k, a.get(k), b.get(k)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-root", type=str, default="experiments/artifacts")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--sort-by", type=str, default="created_at", help="Metric to sort by (max_drawdown, sharpe, created_at, current_mse)")
    p.add_argument("--desc", action="store_true")
    p.add_argument("--export", type=str, default=None, help="Export format: csv,tsv,json")
    p.add_argument("--export-path", type=str, default=None, help="Where to write exported file")
    p.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="Compare two run IDs (or paths)")
    args = p.parse_args()

    runs = _collect_runs(args.artifacts_root)
    if not runs:
        print("no runs found in", args.artifacts_root)
        return

    key = args.sort_by

    def _key(r):
        v = r.get(key)
        try:
            return float(v)
        except Exception:
            return r.get("created_at") or ""

    runs = sorted(runs, key=_key, reverse=bool(args.desc))

    # if compare requested, try to locate both runs by id or path
    if args.compare:
        a_id, b_id = args.compare
        def _find(id_: str) -> Optional[dict]:
            # match run_id or run_dir suffix
            for r in runs:
                if r.get("run_id") == id_ or r.get("run_dir", "").endswith(id_):
                    return r
            # try direct path
            if os.path.isdir(id_):
                sfn = os.path.join(id_, "summary.json")
                if os.path.exists(sfn):
                    try:
                        with open(sfn, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                            data.setdefault("run_dir", id_)
                            data.setdefault("run_id", os.path.basename(id_))
                            return data
                    except Exception:
                        return None
            return None

        a = _find(a_id)
        b = _find(b_id)
        if a is None or b is None:
            print("could not find one or both runs to compare")
            return
        table = compare_two_runs(a, b)
        # print compact table
        colw = max(len(str(x)) for row in table for x in row)
        print("Comparison:")
        for k, va, vb in table:
            print(f"{k}:\n  A: {va}\n  B: {vb}\n")
        # optionally export
        if args.export and args.export_path:
            # write JSON for comparison convenience
            if args.export.lower() == "json":
                with open(args.export_path, "w", encoding="utf-8") as fh:
                    json.dump({"A": a, "B": b}, fh, indent=2)
            else:
                # CSV/TSV
                delim = "," if args.export.lower() == "csv" else "\t"
                keys = [r[0] for r in table]
                with open(args.export_path, "w", encoding="utf-8", newline="") as fh:
                    import csv as _csv

                    w = _csv.writer(fh, delimiter=delim)
                    w.writerow(["key", "A", "B"])
                    for k, va, vb in table:
                        w.writerow([k, va, vb])
        return

    # normal listing
    print(f"Showing {min(len(runs), args.limit)} runs from {args.artifacts_root}")
    for r in runs[: args.limit]:
        print("-" * 60)
        print("run_id:", r.get("run_id"))
        print("run_dir:", r.get("run_dir"))
        print("created_at:", r.get("created_at"))
        print("passed:", r.get("passed"))
        print("max_drawdown:", r.get("max_drawdown"))
        print("sharpe:", r.get("sharpe"))
        print("current_mse:", r.get("current_mse"))
        print("baseline_mse:", r.get("baseline_mse"))
        print("n_executions:", r.get("n_executions"))

    if args.export and args.export_path:
        export_runs(runs[: args.limit], args.export_path, fmt=args.export)


if __name__ == "__main__":
    main()
