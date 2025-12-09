#!/usr/bin/env python3
"""Backfill Kraken OHLC and write immutable parquet snapshots under data/raw.

Usage:
  python tooling/backfill_kraken.py --symbols XBT/USD,ETH/USD --interval 1m --months 6

This script imports the project's `src` modules (so run from repo root).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import os
import sys
import traceback

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", type=str, required=True, help="Comma-separated list of pairs (e.g. XBT/USD,ETH/USD)")
    p.add_argument("--interval", type=str, default="1m")
    p.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (defaults to now - months)")
    p.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (defaults to today)")
    p.add_argument("--months", type=int, default=6, help="How many months back to fetch when --start not provided")
    return p.parse_args()


def dt_to_epoch_seconds(dt: datetime) -> int:
    return int(dt.replace(tzinfo=None).timestamp())


def main():
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    if args.end:
        end_dt = datetime.fromisoformat(args.end)
    else:
        end_dt = datetime.utcnow()
    if args.start:
        start_dt = datetime.fromisoformat(args.start)
    else:
        start_dt = end_dt - timedelta(days=int(args.months * 30))

    start_ts = dt_to_epoch_seconds(start_dt)
    end_ts = dt_to_epoch_seconds(end_dt)

    print(f"Backfill: symbols={symbols} interval={args.interval} start={start_dt.date()} end={end_dt.date()}")

    # Ensure repo root is on sys.path so imports like `src.ingestion.providers` work
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from src.ingestion.providers import kraken_rest
        from src.ingestion.readers import snapshot_to_parquet
    except Exception as e:
        print("Failed to import repo modules:", e)
        traceback.print_exc()
        sys.exit(2)

    for sym in symbols:
        print(f"Fetching {sym} from {start_dt.isoformat()} to {end_dt.isoformat()}")
        try:
            df = kraken_rest.get_ohlc(sym, args.interval, since=start_ts, end=end_ts)
            if df is None or df.empty:
                print(f"No data returned for {sym}")
                continue
            # write snapshot (reader will validate & partition by date)
            out = snapshot_to_parquet(sym, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), df, out_root="data/raw")
            print(f"Wrote snapshot for {sym}: {out}")
        except Exception as e:
            print(f"Error fetching/writing for {sym}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
