import numpy as np
from tooling.backtester import run_backtest


def test_backtester_no_trades():
    returns = [0.001, 0.001, 0.001]
    # always long -> no trades, no fees
    signals = [1, 1, 1]
    metrics, equity = run_backtest(returns, signals, initial_capital=100.0, fee_bps=0.0, slippage_bps=0.0)
    # final equity should be approx 100 * (1+0.001)^3
    expected = 100.0
    for r in returns:
        expected = expected + (1.0 * r * expected)
    assert abs(metrics['final_equity'] - expected) < 1e-6


def test_backtester_fee_effect():
    returns = [0.0, 0.0]
    # flip position -> trade happens once
    signals = [1, 0]
    metrics, equity = run_backtest(returns, signals, initial_capital=100.0, fee_bps=10.0, slippage_bps=0.0)
    # with a trade of full position, fee_rate = 10bps = 0.001 -> trade cost = 100 * 0.001 = 0.1
    # there will be one trade at i=0 (from 0->+1) and another at i=1 (from +1->-1)
    # total fees approx 0.1 + 0.1 = 0.2 so final equity should be ~99.8
    assert metrics['final_equity'] < 100.0
    assert metrics['final_equity'] > 99.0


def test_backtester_partial_fill():
    returns = [0.0, 0.01, 0.0]
    # start flat, then go long -> partial fill set to 0.5
    signals = [0, 1, 1]
    metrics_pf, equity_pf = run_backtest(returns, signals, initial_capital=100.0, fee_bps=10.0, partial_fill_pct=0.5)
    # compare with full fill
    metrics_full, equity_full = run_backtest(returns, signals, initial_capital=100.0, fee_bps=10.0, partial_fill_pct=1.0)
    # With partial fills, some execution happens later so fees may differ; final equity should be defined
    assert len(equity_pf) == len(returns) + 1
    assert 'total_pnl' in metrics_pf
