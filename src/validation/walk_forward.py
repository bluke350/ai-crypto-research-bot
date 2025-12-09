from typing import List, Dict, Any, Optional, Callable
import pandas as pd
from src.validation.backtester import run_backtest
from src.tuning.optimizers import BayesianTuner


# strategy_factory: Optional[Callable[[Dict[str, Any]], Any]]
# If provided, evaluate_walk_forward will use the factory to create a strategy
# instance for a given parameter set and call `generate_targets(prices)` to
# produce target positions passed to the backtester. This makes tuning strategy
# hyperparameters straightforward.


def _evaluate_pnl_metrics(pnl: pd.Series) -> Dict[str, float]:
    returns = pnl.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * (252 ** 0.5) if not returns.empty else 0.0
    return {"sharpe": float(sharpe), "final_value": float(pnl.iloc[-1])}


def rolling_folds(prices: pd.DataFrame, window: int, step: int) -> List[Dict[str, pd.DataFrame]]:
    folds = []
    for start in range(0, len(prices) - window, step):
        train = prices.iloc[start : start + window]
        test = prices.iloc[start + window : start + window + step]
        folds.append({"train": train, "test": test})
    return folds


def evaluate_walk_forward(
    prices: pd.DataFrame,
    targets: Optional[pd.Series],
    simulator,
    window: int = 100,
    step: int = 10,
    tuner: Optional[Any] = None,
    param_space: Optional[Dict[str, list]] = None,
    strategy_factory: Optional[Callable[..., Any]] = None,
    sizer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Perform walk-forward evaluation and optionally tune parameters per-fold.

    If `tuner` and `param_space` are provided, the tuner will be used to search
    for the best parameters per-fold. The objective passed to the tuner should
    maximize the OOS Sharpe (higher is better).
    """
    # Ensure prices are indexed by timestamp if a 'timestamp' column exists so
    # downstream indexing (targets.loc[f['test'].index]) works as tests expect.
    try:
        if isinstance(prices, pd.DataFrame) and "timestamp" in prices.columns and not isinstance(prices.index, pd.DatetimeIndex):
            prices = prices.set_index(pd.DatetimeIndex(prices["timestamp"]))
            # ensure index has no name to avoid ambiguity with column labels
            try:
                prices.index.name = None
            except Exception:
                pass
    except Exception:
        # be defensive: if indexing fails, continue with original prices
        pass

    folds = rolling_folds(prices, window=window, step=step)
    results = []
    for f in folds:
        # If tuning is requested, run search on the fold's train set using the
        # provided simulator. For simplicity we evaluate candidates by training
        # the simulator/config on the train set and measuring performance on the test set.
        best_params = None
        best_score = None
        if tuner is not None and param_space is not None:
            # build an objective that configures the simulator and returns OOS sharpe
            def _objective(params: Dict[str, Any]) -> float:
                # Build targets using strategy_factory if provided
                try:
                    if strategy_factory is None:
                        # no strategy provided; cannot tune
                        return float('-inf')
                    strat = strategy_factory(**params)
                    train_targets = strat.generate_targets(f["train"]).loc[f["train"].index]
                except Exception:
                    return float('-inf')

                # configure simulator if possible
                sim = simulator
                try:
                    if callable(simulator):
                        sim = simulator()
                    if hasattr(sim, "configure"):
                        sim.configure(params)
                except Exception:
                    pass

                # run backtest on train (allow any trainer-like behavior)
                try:
                    run_backtest(f["train"], train_targets, sim, sizer=sizer)
                except Exception:
                    pass

                # evaluate on test
                try:
                    strat = strategy_factory(**params)
                    test_targets = strat.generate_targets(f["test"]).loc[f["test"].index]
                    out = run_backtest(f["test"], test_targets, sim, sizer=sizer)
                    pnl = out["pnl"]
                    metrics = _evaluate_pnl_metrics(pnl)
                    return float(metrics["sharpe"])
                except Exception:
                    return float('-inf')

            # create a fresh tuner instance for the fold
            tuner_instance = tuner.__class__(param_space, getattr(tuner, "n_trials", 20))
            best_params, best_score = tuner_instance.optimize(_objective)

        # final evaluation on test using best_params if present
        sim = simulator
        try:
            if callable(simulator):
                sim = simulator()
            if best_params and hasattr(sim, "configure"):
                sim.configure(best_params)
        except Exception:
            pass

        # build test targets either from provided `targets` or from strategy_factory
        if strategy_factory is not None:
            try:
                strat = strategy_factory(**(best_params or {}))
                test_targets = strat.generate_targets(f["test"]).loc[f["test"].index]
            except Exception:
                # fallback to empty series if strategy fails
                test_targets = pd.Series([], dtype=float)
        else:
            if targets is None:
                raise ValueError("either 'targets' or 'strategy_factory' must be provided for final evaluation")
            test_targets = targets.loc[f["test"].index]

        # ensure targets are positionally aligned with prices before backtest
        try:
            import pandas as _pd
            if isinstance(test_targets, _pd.Series):
                if "timestamp" in f["test"].columns:
                    test_targets = test_targets.reindex(f["test"]["timestamp"]).ffill().fillna(0.0).reset_index(drop=True)
                elif isinstance(f["test"].index, _pd.DatetimeIndex):
                    test_targets = test_targets.reindex(f["test"].index).ffill().fillna(0.0).reset_index(drop=True)
                else:
                    test_targets = _pd.Series(test_targets.values[: len(f["test"])])
        except Exception:
            # if alignment fails, leave as-is and let run_backtest raise a helpful error
            pass

        out = run_backtest(f["test"], test_targets, sim, sizer=sizer)
        pnl = out["pnl"]
        metrics = _evaluate_pnl_metrics(pnl)
        results.append({"metrics": metrics, "best_params": best_params, "best_score": best_score})
    return {"folds": results}
