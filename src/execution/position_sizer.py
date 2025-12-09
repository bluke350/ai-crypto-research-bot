from __future__ import annotations
import math
from collections import deque
from typing import Optional

import numpy as np


class VolatilityRiskSizer:
    """Volatility-aware position sizer that turns direction signals into unit sizes.

    The sizer targets a fixed fraction of equity as risk per trade and uses
    realized volatility to estimate a stop distance. Size is capped by
    leverage/position limits and rounded to the exchange lot size.
    """

    def __init__(
        self,
        risk_fraction: float = 0.01,
        vol_lookback: int = 30,
        vol_floor: float = 1e-4,
        stop_multiple: float = 1.5,
        max_leverage: float = 2.0,
        max_position_fraction: float = 1.0,
        max_notional: Optional[float] = None,
        min_notional: float = 0.0,
        lot_size: float = 1e-6,
        signal_scales_risk: bool = True,
    ) -> None:
        if vol_lookback <= 0:
            raise ValueError("vol_lookback must be positive")
        self.risk_fraction = float(risk_fraction)
        self.vol_lookback = int(vol_lookback)
        self.vol_floor = float(vol_floor)
        self.stop_multiple = float(stop_multiple)
        self.max_leverage = float(max_leverage)
        self.max_position_fraction = float(max_position_fraction)
        self.max_notional = max_notional
        self.min_notional = float(min_notional)
        self.lot_size = float(lot_size)
        self.signal_scales_risk = signal_scales_risk

        self._returns = deque(maxlen=self.vol_lookback)
        self._last_price: Optional[float] = None

    def observe(self, price: float) -> None:
        """Update internal volatility estimate with the latest price."""
        if price is None or price <= 0:
            return
        if self._last_price is not None and self._last_price > 0:
            ret = math.log(price / self._last_price)
            self._returns.append(ret)
        self._last_price = price

    def _realized_vol(self) -> float:
        if not self._returns:
            return self.vol_floor
        vol = float(np.std(self._returns))
        return max(vol, self.vol_floor)

    def size(
        self,
        target_signal: float,
        price: float,
        equity: float,
        current_position: float = 0.0,
    ) -> float:
        """Return desired position (units) given signal and account state."""
        if price <= 0 or equity <= 0:
            return 0.0

        # signal represents desired direction/strength; zero -> flat
        strength = abs(float(target_signal))
        if strength == 0:
            return 0.0
        strength = min(strength, 1.0)

        risk = self.risk_fraction * equity
        if self.signal_scales_risk:
            risk *= strength

        vol = self._realized_vol()
        stop_dist = max(vol * price * self.stop_multiple, price * self.vol_floor)
        if stop_dist == 0:
            return 0.0

        desired_units = risk / stop_dist
        desired_notional = desired_units * price

        # cap by leverage/position fraction and explicit notional cap
        max_notional_leverage = equity * self.max_leverage
        max_notional_pos_frac = equity * self.max_position_fraction
        cap_notional = min(max_notional_leverage, max_notional_pos_frac)
        if self.max_notional is not None:
            cap_notional = min(cap_notional, float(self.max_notional))

        desired_notional = min(desired_notional, cap_notional)
        desired_notional = max(desired_notional, self.min_notional)

        # convert back to units and align to lot size
        units = desired_notional / price
        if self.lot_size > 0:
            units = math.floor(units / self.lot_size) * self.lot_size

        # preserve direction
        units = math.copysign(units, target_signal)

        # avoid churn: if the suggested change is negligible, keep current
        if abs(units - current_position) < self.lot_size:
            return current_position
        return units
