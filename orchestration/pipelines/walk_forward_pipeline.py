from __future__ import annotations

import argparse
import uuid
import json
import os
import logging
from typing import Optional

import pandas as pd

from src.utils.config import load_yaml
from src.validation.walk_forward import evaluate_walk_forward
from src.execution.simulator import Simulator
from src.strategies.moving_average import MovingAverageCrossover
from src.ingestion.providers import kraken_rest

try:
    # optional adapter only required when evaluating RL policies
    from src.models.rl.policy_adapter import RLPolicyStrategy
except Exception:
    RLPolicyStrategy = None


LOG = logging.getLogger(__name__)


def load_prices_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default=None)
    p.add_argument("--prices-csv", type=str, default=None)
    p.add_argument("--ppo-checkpoint", type=str, default=None, help="Path to PPO checkpoint to evaluate (optional)")
    p.add_argument("--window", type=int, default=120)
    p.add_argument("--step", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="experiments/artifacts")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    run_id = str(uuid.uuid4())
    out_dir = os.path.join(args.output, run_id)
    os.makedirs(out_dir, exist_ok=True)

    cfg = {}
    try:
        cfg = load_yaml("configs/project.yaml").get("project", {})
    except Exception:
        # best effort; not fatal
        pass

    symbol = args.symbol
    if symbol is None:
        try:
            kcfg = load_yaml("configs/kraken.yaml").get("kraken", {})
            syms = kcfg.get("symbols", [])
            symbol = syms[0] if syms else "XBT/USD"
        except Exception:
            symbol = "XBT/USD"

    # load price history
    prices: Optional[pd.DataFrame] = None
    if args.prices_csv:
        prices = load_prices_from_csv(args.prices_csv)
    else:
        try:
            # attempt to fetch via Kraken REST (since=0 -> best-effort full history)
            prices = kraken_rest.get_ohlc(symbol, "1m", since=0)
        except Exception as e:
            LOG.warning("Failed to load OHLC from kraken REST: %s", e)
            raise

    # pick strategy factory
    if args.ppo_checkpoint:
        if RLPolicyStrategy is None:
            raise RuntimeError("RLPolicyStrategy adapter not available (torch required)")
        # create a factory that will be called by evaluate_walk_forward(**params)
        def strategy_factory(**params):
            # users may tune `window` or `size` via param_space
            return RLPolicyStrategy(ckpt_path=args.ppo_checkpoint, **params)
    else:
        strategy_factory = MovingAverageCrossover

    # simulator factory
    sim_factory = lambda: Simulator(seed=args.seed)

    res = evaluate_walk_forward(prices=prices, targets=None, simulator=sim_factory, window=args.window, step=args.step, strategy_factory=strategy_factory)

    # dump results
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print(f"Walk-forward completed; artifacts in {out_dir}")


if __name__ == "__main__":
    main()
