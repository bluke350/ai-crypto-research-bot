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
from src.execution.position_sizer import VolatilityRiskSizer
from src.strategies.moving_average import MovingAverageCrossover
from src.ingestion.providers import kraken_rest

LOG = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default=None)
    p.add_argument("--prices-csv", type=str, default=None)
    p.add_argument("--dedupe", type=str, default="first", choices=["none", "first", "last", "mean"], help="How to handle duplicate timestamps in CSV: none|first|last|mean")
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
    p.add_argument("--disable-auto-size", action="store_true", help="Disable auto position sizing; interpret strategy outputs as raw units")
    p.add_argument("--sizer-risk-fraction", type=float, default=0.01, help="Fraction of equity to risk per trade when auto sizing (default 1%)")
    p.add_argument("--sizer-vol-lookback", type=int, default=30, help="Lookback window for realized vol used in sizing")
    p.add_argument("--sizer-stop-multiple", type=float, default=1.5, help="Multiple of vol*price to approximate stop distance")
    p.add_argument("--sizer-max-leverage", type=float, default=2.0, help="Cap notional at this leverage on equity")
    p.add_argument("--sizer-max-position-fraction", type=float, default=1.0, help="Cap notional at this fraction of equity")
    p.add_argument("--sizer-lot-size", type=float, default=1e-6, help="Round position units to this lot size")
    p.add_argument("--sizer-min-notional", type=float, default=0.0, help="Minimum notional to trade when auto sizing")
    # cost modeling CLI flags (passed through to Simulator factory)
    p.add_argument("--slippage-pct", type=float, default=None, help="Fixed slippage pct to apply to simulated fills (e.g., 0.001)")
    p.add_argument("--fee-pct", type=float, default=None, help="Fixed fee pct to apply to simulated fills (e.g., 0.00075)")
    p.add_argument("--stochastic-costs", action="store_true", help="Enable stochastic slippage/latency so seeds produce divergent cost paths")
    p.add_argument("--latency-base-ms", type=int, default=50, help="Base latency in ms for latency sampler")
    p.add_argument("--latency-jitter-ms", type=int, default=100, help="Jitter in ms for latency sampler")
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
        # use shared CSV loader with dedupe support
        from src.utils.io import load_prices_csv
        prices = load_prices_csv(args.prices_csv, dedupe=args.dedupe)
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

    # build a simulator factory that constructs structured cost models per-seed
    def sim() -> Simulator:
        try:
            from src.execution.cost_models import FeeModel, SlippageModel, LatencySampler
        except Exception:
            FeeModel = None
            SlippageModel = None
            LatencySampler = None
        fee = None
        slip = None
        lat = None
        if 'FeeModel' in globals() or FeeModel is not None:
            try:
                fee = FeeModel(fixed_fee_pct=float(args.fee_pct) if args.fee_pct is not None else None)
            except Exception:
                fee = None
        if 'SlippageModel' in globals() or SlippageModel is not None:
            try:
                slip = SlippageModel(fixed_slippage_pct=float(args.slippage_pct) if args.slippage_pct is not None else None,
                                      stochastic_sigma=0.1 if args.stochastic_costs else 0.0,
                                      seed=int(args.seed))
            except Exception:
                slip = None
        if 'LatencySampler' in globals() or LatencySampler is not None:
            try:
                lat = LatencySampler(base_ms=int(args.latency_base_ms), jitter_ms=int(args.latency_jitter_ms), seed=int(args.seed))
            except Exception:
                lat = None
        return Simulator(seed=args.seed, fee_model=fee, slippage_model=slip, latency_model=lat)

    sizer = None
    try:
        if not getattr(args, "disable_auto_size", False):
            sizer = VolatilityRiskSizer(
                risk_fraction=float(args.sizer_risk_fraction),
                vol_lookback=int(args.sizer_vol_lookback),
                stop_multiple=float(args.sizer_stop_multiple),
                max_leverage=float(args.sizer_max_leverage),
                max_position_fraction=float(args.sizer_max_position_fraction),
                lot_size=float(args.sizer_lot_size),
                min_notional=float(args.sizer_min_notional),
            )
    except Exception:
        LOG.exception("failed to build position sizer; continuing without auto sizing")

    res = evaluate_walk_forward(prices=prices, targets=None, simulator=sim, window=args.window, step=args.step, tuner=tuner, param_space=param_space, strategy_factory=strategy_factory, sizer=sizer)
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
                out = evaluate_walk_forward(prices=sub, targets=None, simulator=sim, window=max(10, args.window//2), step=max(5, args.step//2), tuner=None, param_space=None, strategy_factory=strategy_factory, sizer=sizer)
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
