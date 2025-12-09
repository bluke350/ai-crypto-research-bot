from __future__ import annotations
import argparse
import os
import uuid
import json
import logging
from typing import Optional

import pandas as pd

from src.utils.io import load_prices_csv
from src.utils.config import load_yaml
from src.features.technical import all_technical
from src.features.targets import meta_labels

LOG = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices-csv", type=str, default=None, help="Path to input prices CSV (optional)")
    p.add_argument("--symbol", type=str, default=None, help="Symbol to fetch from REST if CSV not provided")
    p.add_argument("--dedupe", type=str, default="first", choices=["none", "first", "last", "mean"], help="How to dedupe CSV timestamps")
    p.add_argument("--out", type=str, default="experiments/artifacts/features", help="Output folder for feature artifacts")
    p.add_argument("--rv-window", type=int, default=30, help="Realized vol window")
    p.add_argument("--vol-window", type=int, default=30, help="Rolling volume window")
    p.add_argument("--pt", type=float, default=0.01, help="Profit-taking pct for triple-barrier")
    p.add_argument("--sl", type=float, default=0.005, help="Stop-loss pct for triple-barrier")
    p.add_argument("--horizon", type=int, default=60, help="Horizon (bars) for triple-barrier")
    p.add_argument("--config", type=str, default=None, help="Project config yaml (optional)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    run_id = str(uuid.uuid4())
    out_dir = os.path.join(args.out, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # load project config if present
    cfg = {}
    if args.config:
        try:
            cfg = load_yaml(args.config).get("project", {})
        except Exception:
            LOG.exception("failed to load config %s", args.config)

    # load prices
    prices = None
    if args.prices_csv:
        prices = load_prices_csv(args.prices_csv, dedupe=args.dedupe)
    else:
        # try to fallback to REST via kraken if symbol provided
        try:
            from src.ingestion.providers import kraken_rest
            sym = args.symbol or (cfg.get("kraken", {}).get("symbols", [None])[0])
            if sym is None:
                raise RuntimeError("No symbol provided and no config fallback available")
            prices = kraken_rest.get_ohlc(sym, "1m", since=0)
        except Exception:
            LOG.exception("failed to load prices via REST; provide --prices-csv")
            raise

    # build features
    tech = all_technical(prices)
    # select rv/vol windows provided
    # the all_technical uses defaults; user can still access rv columns directly by name

    # build meta labels (triple-barrier) from close
    try:
        labels = meta_labels(prices["close"], pt=args.pt, sl=args.sl, horizon=args.horizon)
    except Exception:
        LOG.exception("failed to compute meta labels")
        labels = pd.DataFrame()

    # join into single dataframe by timestamp (keep original timestamps)
    out_df = prices[["timestamp"]].copy()
    out_df = out_df.merge(tech, on="timestamp", how="left")
    if not labels.empty:
        out_df = out_df.merge(labels, left_index=True, right_index=True, how="left")

    # write artifact
    fout = os.path.join(out_dir, "features.parquet")
    try:
        out_df.to_parquet(fout)
    except Exception:
        # fallback: write csv
        fout_csv = os.path.join(out_dir, "features.csv")
        out_df.to_csv(fout_csv, index=False)
        fout = fout_csv

    meta = {"run_id": run_id, "artifact": fout}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Features written to {out_dir}")


if __name__ == "__main__":
    main()
