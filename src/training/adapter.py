from __future__ import annotations

from typing import Any

import pandas as pd

from src.training.inference import ModelWrapper, load_checkpoint
from src.validation.backtester import run_backtest


def inference_backtest(checkpoint_path: str, prices: pd.DataFrame, simulator: Any = None, *,
                       executor: Any = None, method: str = "threshold", sizing_mode: str = "units", **mapper_kwargs) -> dict:
    """Run inference-to-backtest flow:

    - Loads checkpoint at `checkpoint_path`.
    - Uses `ModelWrapper` to predict returns and map to targets using `method`.
    - Calls `run_backtest` with `prices`, `targets`, and `simulator`.

    Returns the backtest result dict from `run_backtest`.
    """
    if prices is None or len(prices) == 0:
        raise ValueError("prices must be provided")

    wrapper = ModelWrapper.from_file(checkpoint_path)
    targets = wrapper.predicted_to_targets(prices, method=method, **mapper_kwargs)

    # run_backtest expects targets aligned to prices index; ensure alignment
    # Our predicted_to_targets returns a Series aligned to prices.index[2:];
    # we will reindex to prices index by prepending two zeros for the first two rows.
    full_targets = pd.Series(0.0, index=prices.index, name="target")
    full_targets.loc[targets.index] = targets

    # run_backtest supports either a simulator (legacy) or an executor (preferred)
    if executor is not None:
        result = run_backtest(prices.assign(timestamp=prices.index), full_targets, None, executor=executor, sizing_mode=sizing_mode)
    else:
        result = run_backtest(prices.assign(timestamp=prices.index), full_targets, simulator, sizing_mode=sizing_mode)
    return result
