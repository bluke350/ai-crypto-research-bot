from __future__ import annotations

import argparse
import json
import os
import uuid
import logging
from typing import Optional, Dict, Any

import pandas as pd

from src.utils.config import load_yaml
from src.validation.walk_forward import evaluate_walk_forward
from src.tuning.optimizers import BayesianTuner
from src.execution.simulator import Simulator
from src.strategies.moving_average import MovingAverageCrossover
from src.ingestion.providers import kraken_rest

LOG = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default=None)
    p.add_argument("--prices-csv", type=str, default=None)
    p.add_argument("--param-space", type=str, default=None, help="JSON string describing param_space")
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--optimizer", type=str, default="bayes", choices=["bayes", "random", "optuna"], help="Which optimizer to use for tuning")
    p.add_argument("--optuna-db", type=str, default=None, help="Optuna DB URL (sqlite:///path) to use for study storage")
    p.add_argument("--optuna-pruner", type=str, default="median", choices=["median", "asha"], help="Optuna pruner to use when optimizer=optuna")
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--step", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="experiments/artifacts")
    p.add_argument("--per-regime", action="store_true", help="Evaluate best params per volatility regime (high/low)")
    p.add_argument("--register", action="store_true", help="Register run and artifacts in experiments DB")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    run_id = str(uuid.uuid4())
    out_dir = os.path.join(args.output, run_id)
    os.makedirs(out_dir, exist_ok=True)

    symbol = args.symbol
    if symbol is None:
        try:
            cfg = load_yaml("configs/kraken.yaml").get("kraken", {})
            syms = cfg.get("symbols", [])
            symbol = syms[0] if syms else "XBT/USD"
        except Exception:
            symbol = "XBT/USD"

    # load data (csv or REST)
    prices: Optional[pd.DataFrame] = None
    if args.prices_csv:
        prices = pd.read_csv(args.prices_csv)
        if "timestamp" in prices.columns:
            prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    else:
        try:
            prices = kraken_rest.get_ohlc(symbol, "1m", since=0)
        except Exception as e:
            LOG.warning("Failed to load OHLC from kraken REST: %s", e)
            raise

    # load param_space from JSON string; fall back to a small default for MA
    if args.param_space:
        param_space = json.loads(args.param_space)
    else:
        param_space = {"short": [5, 10], "long": [20, 40], "size": [1.0]}

    if args.optimizer == "optuna":
        try:
            from src.tuning.optimizers import OptunaTuner
        except Exception:
            raise RuntimeError("optuna not available; install optuna or choose a different optimizer")
        tuner = OptunaTuner(param_space, n_trials=args.n_trials, seed=args.seed, db_url=args.optuna_db, pruner=args.optuna_pruner)
    elif args.optimizer == "random":
        # Prefer a project-local RandomSearchTuner if present; fall back to
        # a more advanced tuner in src.tuning.optimizers if available.
        try:
            from src.tuning.optimizers import RandomSearchTuner  # type: ignore
        except Exception:
            from src.tuning.simple_tuner import RandomSearchTuner
        tuner = RandomSearchTuner(param_space, n_trials=args.n_trials, seed=args.seed)
    else:
        tuner = BayesianTuner(param_space, n_trials=args.n_trials, seed=args.seed)

    # strategy factory: Moving Average by default
    # strategy factory expects keyword params; wrap the class so it matches the expected callable signature
    def strategy_factory(**params: Dict[str, Any]) -> Any:
        # fill defaults for MovingAverageCrossover
        short = params.get("short", 5)
        long = params.get("long", 20)
        size = params.get("size", 1.0)
        return MovingAverageCrossover(short=short, long=long, size=size)

    sim = lambda: Simulator(seed=args.seed)

    res = evaluate_walk_forward(prices=prices, targets=None, simulator=sim, window=args.window, step=args.step, tuner=tuner, param_space=param_space, strategy_factory=strategy_factory)
    # if per-regime, evaluate best params per regime and attach to res
    if args.per_regime:
        try:
            from src.features.regime import identify_regimes
            lbls = identify_regimes(prices.set_index(pd.DatetimeIndex(prices['timestamp']))['close']) if 'timestamp' in prices.columns else identify_regimes(prices['close'])
            # split prices into regimes and evaluate using same sim and strategy
            per = {}
            for r in sorted(lbls.unique()):
                mask = lbls == r
                if mask.sum() < 10:
                    continue
                sub = prices.loc[mask.index[mask].tolist()]
                out = evaluate_walk_forward(prices=sub, targets=None, simulator=sim, window=max(10, args.window//2), step=max(5, args.step//2), tuner=None, param_space=None, strategy_factory=strategy_factory)
                per[int(r)] = out
            res['per_regime'] = per
        except Exception:
            LOG.exception('failed to compute per-regime evaluation')

    # Register results if requested
    if args.register:
        from src.persistence.db import RunLogger
        run_id = str(uuid.uuid4())
        with RunLogger(run_id, cfg={"symbol": symbol, "param_space": param_space}):
            out_path = os.path.join(out_dir, 'results.json')
            try:
                import json as _json
                _json.dump(res, open(out_path, 'w', encoding='utf-8'), indent=2)
            except Exception:
                LOG.exception('failed to write tuning results')
            try:
                with RunLogger(run_id, cfg={}) as rl:
                    rl.log_artifact(out_path, kind='tuning_results')
            except Exception:
                LOG.exception('failed to register tuning artifact')

    import json as _json
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        _json.dump(res, fh, indent=2)

    print(f"Tuning completed; artifacts in {out_dir}")


if __name__ == "__main__":
    main()
