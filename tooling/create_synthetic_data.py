"""Create synthetic minute-parquet price data for a pair.

Writes a single parquet file under:
  <data_root>/<pair>/<YYYYMMDD>/<YYYYMMDDTHHMM>.parquet

Columns: timestamp (UTC), close (float)
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd


def make_walk(start_price: float, n: int, sigma: float = 0.001):
    # geometric random-walk
    returns = np.random.normal(loc=0.0, scale=sigma, size=n)
    prices = start_price * np.exp(np.cumsum(returns))
    return prices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", required=True)
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--minutes", type=int, default=120)
    p.add_argument("--start-price", type=float, default=30000.0)
    args = p.parse_args()

    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = now - timedelta(minutes=args.minutes - 1)
    timestamps = [start + timedelta(minutes=i) for i in range(args.minutes)]
    prices = make_walk(args.start_price, args.minutes)

    df = pd.DataFrame({"timestamp": timestamps, "close": prices})

    outdir = os.path.join(args.data_root, args.pair, start.strftime("%Y%m%d"))
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, start.strftime("%Y%m%dT%H%M") + ".parquet")
    df.to_parquet(fname)
    print(f"wrote {fname} with {len(df)} rows")


if __name__ == "__main__":
    main()
