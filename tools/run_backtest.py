"""Small utility to run a synthetic backtest locally and save executions.

Usage: python tools/run_backtest.py
"""
import pandas as pd
from src.execution.simulator import Simulator
from src.validation.backtester import run_backtest


def make_price_series(n=60, start_price=100.0):
    times = pd.date_range("2022-01-01", periods=n, freq="min")
    prices = pd.DataFrame({"timestamp": times, "close": [start_price + (i * 0.1) for i in range(n)]})
    return prices


def main():
    sim = Simulator(rules={"partial_fill_fraction": 0.75, "slippage_daily_vol": 0.3, "slippage_k": 0.05})
    prices = make_price_series()
    # simple target: go to +1 unit at t=10, back to 0 at t=40
    targets = pd.Series([0.0] * len(prices))
    targets.iloc[10] = 1.0
    targets.iloc[40] = 0.0

    out = run_backtest(prices, targets, sim)
    execs = out["executions"]
    if not execs.empty:
        execs.to_csv("backtest_executions.csv", index=False)

    print("Backtest complete")
    print(f"Final portfolio value: {out['pnl'].iloc[-1]:.2f}")
    print(f"Total fees: {out['stats'].get('fees', 0):.6f}")
    print(f"Executions saved to backtest_executions.csv ({len(execs)} rows)")


if __name__ == '__main__':
    main()
