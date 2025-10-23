from __future__ import annotations
import pandas as pd
from typing import Dict, Any
from src.execution.order_models import Order


def run_backtest(prices: pd.DataFrame, targets: pd.Series, simulator, initial_cash: float = 1_000_000.0) -> Dict[str, Any]:
    """Run a simple backtest that executes market orders to reach target position units.

    Parameters
    ----------
    prices: DataFrame
        Must contain 'timestamp' and 'close' columns sorted by timestamp.
    targets: Series
        Target position units aligned to prices (same index). E.g., -1.0..1.0 representing position units.
    simulator: Simulator
        Simulator instance with place_order(order, market_price) method.
    initial_cash: float
        Starting cash.

    Returns
    -------
    dict
        { 'pnl': Series of portfolio value over time, 'executions': DataFrame of fills, 'stats': dict }
    """
    if prices is None or len(prices) == 0:
        raise ValueError("prices must not be empty")
    prices = prices.sort_values("timestamp").reset_index(drop=True)
    if len(prices) != len(targets):
        raise ValueError("prices and targets must align and have same length")

    cash = float(initial_cash)
    position = 0.0
    pnl = []
    exec_rows = []

    for idx, row in prices.iterrows():
        price = float(row["close"])
        target = float(targets.iat[idx])
        delta = target - position
        if abs(delta) > 1e-12:
            side = "buy" if delta > 0 else "sell"
            ord_obj = Order(order_id=f"o{idx}", pair="XBT/USD", side=side, size=delta, price=None)
            fill = simulator.place_order(ord_obj, market_price=price, is_maker=False)
            fee = fill.get("fee", 0.0)
            # update cash and position
            cash -= (fill.get("avg_fill_price", price) * delta) + fee
            position += delta
            exec_row = dict(fill)
            exec_row["timestamp"] = row["timestamp"]
            exec_rows.append(exec_row)

        # mark-to-market portfolio value
        value = cash + position * price
        pnl.append(value)

    pnl_series = pd.Series(pnl, index=prices["timestamp"]).rename("portfolio_value")
    exec_df = pd.DataFrame(exec_rows)
    stats = getattr(simulator, "stats", {})
    return {"pnl": pnl_series, "executions": exec_df, "stats": stats}
