"""Simple production-style backtester utilities.

Provides a small, deterministic simulator for binary-signal strategies that
supports per-trade fees, slippage, and position sizing. Designed to be
self-contained so `tooling/canary_promote.py` can use it to produce
realistic promotion metrics.

The model assumptions (simplified):
- Input: per-bar returns in decimal (e.g., 0.001 = 0.1%) contained in a numpy array.
- Signal: array-like of 0/1 predictions aligned with returns.
- Execution: when signal changes, we pay fees+slippage proportional to traded
  notional. We apply the position for the bar's return (i.e., pos * return).
- Position mapping: signal==1 -> +1, signal==0 -> -1 (short). You can change
  this convention by pre-processing signals before calling run_backtest.

This is intentionally small — for production you may want a more advanced
engine that models price ladders, partial fills, per-exchange fees, and
execution delay.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def run_backtest(
    returns: Iterable[float],
    signals: Iterable[int],
    initial_capital: float = 1.0,
    fee_bps: float = 5.0,
    slippage_bps: float = 10.0,
    position_size: float = 1.0,
    fixed_fee: float = 0.0,
    partial_fill_pct: float = 1.0,
) -> Tuple[dict, np.ndarray]:
    """Run a simple backtest and return metrics + equity curve.

    Args:
        returns: array-like of per-bar returns (decimal). e.g., 0.001 for 0.1%.
        signals: array-like of 0/1 decisions for each bar.
        initial_capital: starting capital (float).
        fee_bps: per-trade fees in basis points (bps).
        slippage_bps: per-trade slippage in bps.
        position_size: fraction of capital to allocate to the notional each trade.

    Returns:
        (metrics_dict, equity_curve_array)
    """
    ret = np.asarray(returns, dtype=float)
    sig = np.asarray(signals, dtype=int)
    n = len(ret)
    if len(sig) != n:
        raise ValueError("returns and signals must have same length")

    # map 1->+1, 0->-1
    pos = np.where(sig == 1, 1.0, -1.0)

    equity = np.empty(n + 1, dtype=float)
    equity[0] = float(initial_capital)

    fee_rate = (fee_bps + slippage_bps) / 10000.0

    # track last position for trade detection
    last_pos = 0.0
    # track any pending remainder of a partial fill (signed position remaining to execute)
    pending_exec = 0.0
    for i in range(n):
        target_pos = pos[i] * position_size
        # desired change in position
        delta = target_pos - last_pos

        exec_amt = 0.0
        # if there's a pending execution from previous partial fill, execute it first
        if pending_exec != 0.0:
            # execute pending amount now
            exec_now = pending_exec
            trade_amt_pending = abs(exec_now) * equity[i]
            trade_cost_pending = trade_amt_pending * fee_rate + fixed_fee
            equity[i] -= trade_cost_pending
            pending_exec = 0.0
            last_pos += exec_now
            # note: do not apply pnl for pending execution until position is in place for the bar

        # execute current trade partially according to partial_fill_pct
        if delta != 0.0:
            exec_now = delta * partial_fill_pct
            # remaining portion will be queued as pending_exec (to be executed next bar)
            pending_exec = delta - exec_now
            trade_amt = abs(exec_now) * equity[i]
            trade_cost = trade_amt * fee_rate + fixed_fee
            equity[i] -= trade_cost
            last_pos += exec_now

        # apply market return for the position held during this bar (last_pos)
        pnl = last_pos * ret[i] * equity[i]
        equity[i + 1] = equity[i] + pnl

    # compute metrics
    strat_returns = np.diff(equity) / equity[:-1]
    total_pnl = float(equity[-1] - initial_capital)
    mean = float(np.nanmean(strat_returns))
    std = float(np.nanstd(strat_returns))
    sharpe = None
    if std and std > 0:
        # simple Sharpe annualized assuming 252 periods (caller should adapt)
        sharpe = float(mean / std * (252 ** 0.5))

    cum = np.cumsum(np.insert(strat_returns, 0, 0.0)) + 1.0
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / (peak + 1e-12)
    max_dd = float(drawdown.min())

    win_rate = float((strat_returns > 0).mean()) if len(strat_returns) > 0 else 0.0

    metrics = {
        "total_pnl": total_pnl,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "final_equity": float(equity[-1]),
    }

    return metrics, equity


if __name__ == "__main__":
    # tiny smoke test
    r = [0.001, -0.002, 0.003, 0.0]
    s = [1, 1, 0, 0]
    m, e = run_backtest(r, s)
    print(m)
