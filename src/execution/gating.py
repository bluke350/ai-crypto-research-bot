from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from src.execution.order_models import Order


@dataclass
class RiskConfig:
    max_position_abs: float = 10.0  # maximum absolute position per account (BTC units)
    max_order_notional: float = 10000.0  # maximum notional per order in USD
    daily_loss_limit: float = 5000.0  # realized loss threshold to trip circuit breaker
    per_symbol_max: Optional[Dict[str, float]] = None  # per-symbol caps
    cooling_seconds: int = 300  # cooling period after circuit trip


class RiskGate:
    """Simple risk gating utility.

    Usage:
      gate = RiskGate(config)
      ok, reason = gate.check_order(order, market_price, positions, realized_pnl)
    """

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self.tripped: bool = False
        self.trip_reason: Optional[str] = None

    def check_order(
        self,
        order: Order,
        market_price: float,
        positions: Dict[str, float],
        realized_pnl: float = 0.0,
    ) -> Tuple[bool, Optional[str]]:
        """Return (allowed, reason). If allowed is False, reason explains why."""
        # if gate previously tripped by daily loss, block all
        if realized_pnl <= -abs(self.config.daily_loss_limit):
            self.tripped = True
            self.trip_reason = "daily_loss_limit"
            return False, "daily loss limit exceeded"

        if self.tripped:
            return False, f"circuit_tripped:{self.trip_reason}"

        # check per-order notional
        notional = abs(order.size * market_price)
        if notional > float(self.config.max_order_notional):
            return False, f"order_notional_too_large:{notional}"

        # check per-symbol cap
        sym_cap = None
        if self.config.per_symbol_max and order.pair in self.config.per_symbol_max:
            sym_cap = float(self.config.per_symbol_max[order.pair])
        max_pos = sym_cap if sym_cap is not None else float(self.config.max_position_abs)

        current = float(positions.get(order.pair, 0.0))
        projected = current + float(order.size)
        if abs(projected) > max_pos:
            return False, f"position_limit_exceeded:projected={projected},max={max_pos}"

        return True, None

    def reset(self) -> None:
        self.tripped = False
        self.trip_reason = None
