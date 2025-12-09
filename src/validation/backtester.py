from __future__ import annotations
import pandas as pd
from typing import Dict, Any, Optional
from src.execution.order_models import Order


def run_backtest(
    prices: pd.DataFrame,
    targets: pd.Series,
    simulator=None,
    *,
    executor=None,
    initial_cash: float = 1_000_000.0,
    sizing_mode: str = "units",
    sizer: Optional[Any] = None,
) -> Dict[str, Any]:
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
    sizer: object, optional
        Optional position sizer with `size(target_signal, price, equity, current_position)` that
        converts direction/strength targets into unit sizes automatically.

    Returns
    -------
    dict
        { 'pnl': Series of portfolio value over time, 'executions': DataFrame of fills, 'stats': dict }
    """
    if prices is None or len(prices) == 0:
        raise ValueError("prices must not be empty")
    prices = prices.sort_values("timestamp").reset_index(drop=True)

    # best-effort: allow targets to be indexed by timestamps (DatetimeIndex)
    # and reindex them to the prices' timestamp column. This makes the
    # backtester resilient to callers that provide datetime-indexed Series.
    try:
        import pandas as _pd
        if isinstance(targets, _pd.Series):
            # if prices has a timestamp column, use it for alignment
            if "timestamp" in prices.columns and not isinstance(targets.index, _pd.RangeIndex):
                try:
                    # align targets to the prices' timestamp values
                    aligned = targets.reindex(prices["timestamp"]).fillna(method="ffill").fillna(0.0)
                    targets = aligned.reset_index(drop=True)
                except Exception:
                    # fall back to positional handling below
                    pass
            elif isinstance(prices.index, _pd.DatetimeIndex) and not isinstance(targets.index, _pd.RangeIndex):
                try:
                    aligned = targets.reindex(prices.index).fillna(method="ffill").fillna(0.0)
                    targets = aligned.reset_index(drop=True)
                except Exception:
                    pass
    except Exception:
        # ignore alignment failures and validate by length below
        pass

    if len(prices) != len(targets):
        raise ValueError("prices and targets must align and have same length")

    # Ensure prices are positive
    if (prices["close"] <= 0).any():
        raise ValueError("prices must be positive")

    cash = float(initial_cash)
    position = 0.0
    pnl = []
    exec_rows = []

    for idx, row in prices.iterrows():
        price = float(row["close"])
        target = float(targets.iat[idx])
        equity = cash + position * price

        # optional volatility-aware sizing to reduce manual sizing decisions
        desired_position = position
        if sizer is not None:
            try:
                if hasattr(sizer, "observe"):
                    sizer.observe(price)
                desired_position = float(sizer.size(target_signal=target, price=price, equity=equity, current_position=position))
            except Exception:
                desired_position = position
            delta = desired_position - position
        else:
            delta = target - position

        if abs(delta) > 1e-12:
            # Determine order size based on sizing_mode (ignored when sizer provided)
            if sizer is not None:
                order_size = delta
            elif sizing_mode == "units":
                order_size = delta
            elif sizing_mode == "notional":
                order_size = delta / price if price != 0 else 0.0
            else:
                raise ValueError("unsupported sizing_mode")

            side = "buy" if order_size > 0 else "sell"
            ord_obj = Order(order_id=f"o{idx}", pair="XBT/USD", side=side, size=order_size, price=None)
            # prefer executor if provided (OrderExecutor interface), otherwise fall back to simulator
            if executor is not None:
                fill = executor.execute(ord_obj, market_price=price, is_maker=False)
            elif simulator is not None:
                fill = simulator.place_order(ord_obj, market_price=price, is_maker=False)
            else:
                raise ValueError("Either 'simulator' or 'executor' must be provided to run_backtest")
            fee = fill.get("fee", 0.0)
            filled = float(fill.get("filled_size", order_size))
            avg_price = float(fill.get("avg_fill_price", price))
            # update cash and position using filled size
            cash -= (avg_price * filled) + fee
            position += filled
            exec_row = dict(fill)
            exec_row["requested_size"] = order_size
            exec_row["requested_notional"] = abs(price * order_size)
            exec_row["side"] = side
            exec_row["is_maker"] = False
            exec_row["timestamp"] = row["timestamp"]
            exec_row["equity_before"] = equity
            exec_row["target_position"] = desired_position
            exec_rows.append(exec_row)

        # mark-to-market portfolio value
        value = cash + position * price
        pnl.append(value)

    pnl_series = pd.Series(pnl, index=prices["timestamp"]).rename("portfolio_value")
    exec_df = pd.DataFrame(exec_rows)
    stats = getattr(simulator, "stats", {})
    return {"pnl": pnl_series, "executions": exec_df, "stats": stats}
