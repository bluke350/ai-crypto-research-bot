"""
Pair selector: score candidate USD pairs using recent data and simple statistical signals.

This module scans `data/raw` for pairs, computes a lightweight score per pair using
recent returns (Sharpe-like), volatility, and average volume, and writes a ranked
JSON with top candidates to `experiments/pair_selection.json`.

It's intentionally inexpensive so it can run frequently (e.g., hourly).
"""
from __future__ import annotations

import argparse
import json
import logging
from math import sqrt
from pathlib import Path
from pathlib import Path as _Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
OUT = ROOT / "experiments" / "pair_selection.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pair_selector")


def discover_pairs(data_root: Path) -> List[str]:
    pairs = set()
    for p in data_root.glob("**/bars.parquet"):
        parts = p.parts
        # try to find a two-part folder like XBT/USD
        found = False
        for i in range(len(parts) - 1):
            a = parts[i].upper()
            b = parts[i + 1].upper()
            # match patterns like XBT/USD
            if b == "USD" and a != "DATA" and a != "RAW":
                pairs.add(f"{parts[i]}/{parts[i+1]}")
                found = True
                break
        if found:
            continue
        # fallback: match single folder like XBTUSD -> XBT/USD
        for i, part in enumerate(parts):
            up = part.upper()
            if up == "USD":
                # bare USD (e.g., a path ending with USD folder) is noisy; skip
                continue
            if up.endswith("USD") and len(up) > 3:
                base = part[:-3]
                if base:
                    pairs.add(f"{base}/{up[-3:]}" )
                    break
    return sorted(pairs)


def score_pair(pair: str, lookback_minutes: int = 60 * 24 * 7) -> Dict[str, float]:
    # find latest parquet for pair
    candidate_parquets = []
    for p in DATA_RAW.glob("**/bars.parquet"):
        if pair.replace("/", "") in str(p) or pair in str(p):
            candidate_parquets.append(p)
    if not candidate_parquets:
        raise FileNotFoundError(f"No parquet found for {pair}")
    latest = max(candidate_parquets, key=lambda p: p.stat().st_mtime)
    df = pd.read_parquet(latest)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
    # require close and volume
    if "close" not in df.columns:
        raise ValueError("close column missing")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    # take lookback window
    df_recent = df.tail(lookback_minutes)
    if len(df_recent) < 10:
        return {"pair": pair, "score": 0.0, "n": len(df_recent), "volatility": 0.0, "mean_volume": 0.0, "mean_return": 0.0}

    closes = df_recent["close"].astype(float)
    rets = closes.pct_change().dropna()
    mean_ret = float(rets.mean())
    std_ret = float(rets.std())
    # annualize factor for minutes: sqrt(252 * 24 * 60)
    ann_factor = sqrt(252.0 * 24.0 * 60.0)
    sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else 0.0
    vol = std_ret * ann_factor
    mean_vol = float(df_recent["volume"].astype(float).mean())

    # score: prefer higher sharpe, moderate volatility, and higher volume
    score = float(sharpe) * 0.6 + (mean_vol / (1.0 + mean_vol)) * 0.3 + (1.0 / (1.0 + vol)) * 0.1
    return {"pair": pair, "score": score, "n": len(df_recent), "volatility": vol, "mean_volume": mean_vol, "mean_return": mean_ret, "sharpe": sharpe}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pair_selector")
    # default to last 24 hours (1440 minutes) as requested
    p.add_argument("--lookback-minutes", type=int, default=60 * 24, help="Lookback window in minutes")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args(argv)

    try:
        from src.features.opportunity import OpportunityPredictor, AnomalyDetector, ml_rank_universe
    except Exception:
        OpportunityPredictor = None
        AnomalyDetector = None
        ml_rank_universe = None

    pairs = discover_pairs(DATA_RAW)
    logger.info("Discovered %d pairs", len(pairs))

    scored = []
    for pair in pairs:
        try:
            s = score_pair(pair, lookback_minutes=args.lookback_minutes)
            scored.append(s)
        except Exception as e:
            logger.debug("Skipping %s: %s", pair, e)

    ranked = sorted(scored, key=lambda x: x.get("score", 0.0), reverse=True)
    out = {"created_at": pd.Timestamp.utcnow().isoformat(), "candidates": ranked[: args.top_k]}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Wrote pair selection top %d -> %s", args.top_k, out_path)
    return 0

    # Try ML-based selection if an OpportunityPredictor exists at models/opportunity.pkl
    model_path = ROOT / 'models' / 'opportunity.pkl'
    if ml_rank_universe is not None and model_path.exists():
        try:
            # build price dict for discovered pairs
            pairs = discover_pairs(DATA_RAW)
            logger.info("Discovered %d pairs (ml flow)", len(pairs))
            prices = {}
            for pair in pairs:
                # find latest parquet for pair
                candidate_parquets = [p for p in DATA_RAW.glob('**/bars.parquet') if pair.replace('/', '') in str(p) or pair in str(p)]
                if not candidate_parquets:
                    continue
                latest = max(candidate_parquets, key=lambda p: p.stat().st_mtime)
                df = pd.read_parquet(latest)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                    df = df.sort_values('timestamp').reset_index(drop=True)
                prices[pair] = df[['close'] + (['volume'] if 'volume' in df.columns else [])].copy()

            predictor = None
            detector = None
            try:
                predictor = OpportunityPredictor.load(str(model_path)) if OpportunityPredictor is not None else None
            except Exception:
                predictor = None
            # anomaly detector optional: look for models/opportunity_anom.pkl
            anom_path = ROOT / 'models' / 'opportunity_anom.pkl'
            if AnomalyDetector is not None and anom_path.exists():
                try:
                    detector = AnomalyDetector.load(str(anom_path))
                except Exception:
                    detector = None

            ranked = ml_rank_universe(prices, predictor=predictor, anomaly_detector=detector)[: args.top_k]
            scored = [{'pair': s, 'score': float(score), 'is_normal': bool(ok)} for s, score, ok in ranked]
            out = {"created_at": pd.Timestamp.utcnow().isoformat(), "candidates": scored}
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(out, indent=2))
            logger.info("Wrote ML pair selection top %d -> %s", args.top_k, out_path)
            return 0
        except Exception as e:
            logger.exception("ML pair selection failed, falling back to heuristic: %s", e)

    # fallback to heuristic scoring
    pairs = discover_pairs(DATA_RAW)
    logger.info("Discovered %d pairs", len(pairs))

    scored = []
    for pair in pairs:
        try:
            s = score_pair(pair, lookback_minutes=args.lookback_minutes)
            scored.append(s)
        except Exception as e:
            logger.debug("Skipping %s: %s", pair, e)

    ranked = sorted(scored, key=lambda x: x.get("score", 0.0), reverse=True)
    out = {"created_at": pd.Timestamp.utcnow().isoformat(), "candidates": ranked[: args.top_k]}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Wrote pair selection top %d -> %s", args.top_k, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
