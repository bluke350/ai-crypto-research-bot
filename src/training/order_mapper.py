from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def adjust_targets_to_lot_and_min_notional(targets: pd.Series, prices: pd.Series, *,
                                           lot_size: float = 0.0001,
                                           min_notional: float = 10.0,
                                           price_tick: Optional[float] = None,
                                           sizing_mode: str = "units") -> pd.Series:
    """Adjust target positions to respect lot sizes and minimum notional.

    Parameters
    - targets: desired position units (Series aligned to prices index)
    - prices: Series of market prices aligned to same index
    - lot_size: minimum tradable unit size (units)
    - min_notional: minimum notional required for an order to be placed
    - price_tick: optionally round price to tick (not used for units)
    - sizing_mode: 'units' (targets interpreted as units) or 'notional' (targets are desired notional)

    Returns adjusted targets Series (same index) where values that would result
    in orders with notional below `min_notional` are set to 0. Values are rounded
    to multiples of `lot_size`.
    """
    if not isinstance(targets, pd.Series):
        targets = pd.Series(targets)
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)

    prices = prices.reindex(targets.index)
    adjusted = targets.copy().astype(float)

    # if sizing_mode is notional, convert desired notional -> units
    if sizing_mode == "notional":
        # avoid division by zero
        units = adjusted.values / np.where(prices.values == 0, np.nan, prices.values)
    else:
        units = adjusted.values

    # round units to nearest multiple of lot_size
    if lot_size <= 0:
        rounded = units
    else:
        rounded = np.round(units / lot_size) * lot_size

    # compute notional of the rounded unit
    notional = np.abs(rounded * prices.values)

    # enforce min_notional: if notional < min_notional -> set to 0
    rounded[notional < min_notional] = 0.0

    # final series
    result = pd.Series(rounded, index=targets.index, name=targets.name)
    return result
