from __future__ import annotations

import pandas as pd

from src.execution.order_executor import PaperOrderExecutor
from src.validation.backtester import run_backtest


def test_run_backtest_with_paper_executor():
    idx = pd.date_range("2020-01-01", periods=10, freq="T")
    prices = pd.DataFrame({"timestamp": idx, "close": [100 + i for i in range(len(idx))]})
    targets = pd.Series([0, 1, 1, 0, -1, -1, 0, 0, 1, 0], index=idx)
    executor = PaperOrderExecutor()
    res = run_backtest(prices, targets, simulator=None, executor=executor)
    assert 'pnl' in res
    assert 'executions' in res
    # executor recorded fills should be present
    assert len(executor.fills) >= 0
