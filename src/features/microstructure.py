from __future__ import annotations

from typing import List, Tuple, Dict
import numpy as np
import pandas as pd


def normalize(v: float, eps: float = 1e-9) -> float:
    return float(v) if abs(v) > eps else 0.0


def best_bid_ask(orderbook: Dict[str, List[Tuple[float, float]]]) -> Tuple[float, float]:
    """Return (best_bid_price, best_ask_price) from an orderbook dict with keys 'bids' and 'asks'.

    Each side is a list of (price, size) pairs sorted best-first.
    """
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    best_bid = float(bids[0][0]) if bids else float("nan")
    best_ask = float(asks[0][0]) if asks else float("nan")
    return best_bid, best_ask


def spread(orderbook: Dict[str, List[Tuple[float, float]]]) -> float:
    """Return quoted spread (ask - bid)."""
    b, a = best_bid_ask(orderbook)
    return float(a - b)


def top_n_size(side: List[Tuple[float, float]], n: int) -> float:
    """Sum sizes of top-n price levels for a given side (bids or asks)."""
    return float(sum([s for _, s in (side[:n] if side else [])]))


def bid_ask_imbalance(orderbook: Dict[str, List[Tuple[float, float]]], n: int = 5) -> float:
    """Simple top-n bid-ask imbalance: (B - A) / (B + A), where B/A are summed sizes."""
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    B = top_n_size(bids, n)
    A = top_n_size(asks, n)
    denom = B + A
    return normalize((B - A) / denom) if denom > 0 else 0.0


def weighted_depth_imbalance(orderbook: Dict[str, List[Tuple[float, float]]], n: int = 10) -> float:
    """Compute a price-weighted depth imbalance over top-n levels.

    Weighted by price distance from mid or by raw price depending on use-case.
    Returns (weighted_bids - weighted_asks) / (weighted_bids + weighted_asks).
    """
    bids = orderbook.get("bids", [])[:n]
    asks = orderbook.get("asks", [])[:n]
    if not bids and not asks:
        return 0.0
    # determine mid-price from top levels
    try:
        best_b = bids[0][0]
        best_a = asks[0][0]
        mid = (best_a + best_b) / 2.0
    except Exception:
        mid = 0.0
    wb = 0.0
    for p, s in bids:
        wb += s * max(0.0, (mid - p))
    wa = 0.0
    for p, s in asks:
        wa += s * max(0.0, (p - mid))
    denom = wb + wa
    return normalize((wb - wa) / denom) if denom > 0 else 0.0


def depth_imbalance_by_price_bands(orderbook: Dict[str, List[Tuple[float, float]]], bands: List[float]) -> Dict[str, float]:
    """Aggregate sizes into price-distance bands relative to mid-price.

    `bands` is a list of positive radii (e.g., [0.0, 0.01, 0.02]) representing fractional distance from mid.
    Returns a dict of imbalance per band.
    """
    if not bands:
        return {}
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    if not bids or not asks:
        # If one side missing, return zeros for all bands
        return {f"band_{i}": 0.0 for i in range(len(bands) - 1)}
    best_b, best_a = best_bid_ask(orderbook)
    mid = (best_a + best_b) / 2.0

    band_ims: Dict[str, float] = {}
    # build band edges as absolute price distances
    edges = [mid * (1.0 + b) for b in bands]
    # Compose pandas Series for sizes keyed by price distance
    bid_df = pd.DataFrame(bids, columns=["price", "size"]) if bids else pd.DataFrame(columns=["price", "size"]) 
    ask_df = pd.DataFrame(asks, columns=["price", "size"]) if asks else pd.DataFrame(columns=["price", "size"]) 
    bid_df["dist"] = mid - bid_df["price"]
    ask_df["dist"] = ask_df["price"] - mid

    # For each consecutive band, sum sizes on each side within band
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        bsize = bid_df[(bid_df["price"] >= lo) & (bid_df["price"] < hi)]["size"].sum() if not bid_df.empty else 0.0
        asize = ask_df[(ask_df["price"] >= lo) & (ask_df["price"] < hi)]["size"].sum() if not ask_df.empty else 0.0
        denom = bsize + asize
        band_ims[f"band_{i}"] = normalize((bsize - asize) / denom) if denom > 0 else 0.0
    return band_ims


__all__ = [
    "best_bid_ask",
    "spread",
    "bid_ask_imbalance",
    "weighted_depth_imbalance",
    "depth_imbalance_by_price_bands",
]
