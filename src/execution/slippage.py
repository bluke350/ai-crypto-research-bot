from __future__ import annotations
import math


def sqrt_impact_slippage(notional: float, daily_vol: float = 0.02, k: float = 0.1) -> float:
    """Estimate slippage in price units using a square-root impact model.

    Simplified model: slippage_pct = k * sqrt(|notional|) * daily_vol

    This returns a percentage (e.g., 0.001 == 0.1%).
    """
    if notional == 0:
        return 0.0
    return k * math.sqrt(abs(notional)) * daily_vol
