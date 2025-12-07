from __future__ import annotations

from typing import Any, Dict

import math

from src.execution.order_models import Order
from src.execution.exchange_rules import get_exchange_rules


class ExchangeSimulator:
    """Simple exchange simulator that enforces exchange rules and computes fees.

    Behavior:
    - Rounds order sizes to lot_size multiples (nearest) and rejects (zero-fills)
      orders where resulting notional < min_notional.
    - Computes fee using the ExchangeRules (maker/taker rates).
    - Returns a dict with keys: filled_size, avg_fill_price, fee
    """

    def __init__(
        self,
        pair: str = "XBT/USD",
        *,
        allow_partial: bool = True,
        partial_fill_ratio: float = 0.8,
        partial_fill_std: float = 0.05,
        slippage_pct: float = 0.0,
        fill_model: str = "deterministic",
        seed: int | None = None,
        orderbook: dict | None = None,
    ):
        self.pair = pair
        self.rules = get_exchange_rules().get(pair)
        self.stats: Dict[str, Any] = {"fills": 0}
        self.allow_partial = bool(allow_partial)
        self.partial_fill_ratio = float(partial_fill_ratio)
        self.partial_fill_std = float(partial_fill_std)
        self.slippage_pct = float(slippage_pct)
        self.fill_model = str(fill_model)
        self.seed = None if seed is None else int(seed)
        # optional static orderbook: {"bids": [(price,size),...], "asks": [(price,size),...]}
        # bids should be sorted desc, asks sorted asc. If provided and fill_model=='orderbook',
        # the simulator will match against the supplied book deterministically.
        self.orderbook = orderbook
        # deterministic RNG for stochastic fills (numpy Generator if available)
        try:
            import numpy as _np

            self._rng = _np.random.default_rng(self.seed)
        except Exception:
            import random as _random

            self._rng = _random.Random(self.seed)

    def place_order(self, order: Order, market_price: float, is_maker: bool = False) -> Dict[str, Any]:
        # requested size may be positive (buy) or negative (sell)
        size = float(order.size)
        # align limit price to exchange tick if provided
        if order.price is not None and self.rules and getattr(self.rules, 'price_tick', None):
            tick = float(self.rules.price_tick)
            # round to nearest tick
            order.price = round(float(order.price) / tick) * tick
        # round to lot multiple (floor toward zero to avoid accidental oversizing)
        lot = float(self.rules.lot_size) if self.rules and self.rules.lot_size else 1.0
        if lot > 0:
            # floor toward zero to avoid accidental oversizing: keep sign
            sign = 1.0 if size >= 0 else -1.0
            abs_units = abs(size)
            # number of lots to keep without exceeding requested size
            lots = float(int(abs_units // lot))
            rounded = sign * lots * lot
        else:
            rounded = size
        notional = abs(rounded * float(market_price))
        # enforce min_notional and minimum lot (min order size)
        min_notional = float(self.rules.min_notional) if getattr(self.rules, 'min_notional', None) is not None else 0.0
        min_lot = float(self.rules.lot_size) if getattr(self.rules, 'lot_size', None) is not None else 0.0
        if abs(rounded) < min_lot or notional < min_notional:
            # simulate rejected/ignored order by returning zero fill
            return {"filled_size": 0.0, "avg_fill_price": float(market_price), "fee": 0.0}

        # determine filled size according to configured fill model
        filled = rounded
        if self.allow_partial and abs(rounded) > 0:
            if self.fill_model == "deterministic":
                filled = rounded * float(self.partial_fill_ratio)
            elif self.fill_model == "orderbook" and self.orderbook is not None:
                # deterministic orderbook matching: match against asks for buys, bids for sells
                remaining = abs(rounded)
                filled_acc = 0.0
                weighted_sum = 0.0
                if rounded > 0:
                    # buy order: consume asks (price, size) ascending
                    asks = list(self.orderbook.get('asks', []))
                    for price, depth in asks:
                        take = min(depth, remaining)
                        if take <= 0:
                            continue
                        filled_acc += take
                        weighted_sum += take * float(price)
                        remaining -= take
                        if remaining <= 1e-12:
                            break
                else:
                    # sell order: consume bids (price, size) descending
                    bids = list(self.orderbook.get('bids', []))
                    for price, depth in bids:
                        take = min(depth, remaining)
                        if take <= 0:
                            continue
                        filled_acc += take
                        weighted_sum += take * float(price)
                        remaining -= take
                        if remaining <= 1e-12:
                            break

                # apply sign back
                sign = 1.0 if rounded >= 0 else -1.0
                if filled_acc <= 0:
                    filled = 0.0
                else:
                    avg_price = weighted_sum / filled_acc
                    # set avg price via slippage logic below using avg_price as market_price
                    filled = sign * float(filled_acc)
                    market_price = avg_price
            else:
                # stochastic model: draw a ratio from a normal clipped to (0,1]
                try:
                    # numpy Generator
                    r = float(self._rng.normal(self.partial_fill_ratio, self.partial_fill_std))
                except Exception:
                    # python random
                    r = float(self._rng.gauss(self.partial_fill_ratio, self.partial_fill_std))
                r = max(0.0, min(1.0, r))
                filled = rounded * r

        # simulate slippage: adjust avg fill price by slippage_pct * sign
        sign = 1.0 if filled >= 0 else -1.0
        avg_price = float(market_price) * (1.0 + sign * float(self.slippage_pct)) if self.slippage_pct else float(market_price)

        notional_filled = abs(filled * float(avg_price))
        fee = get_exchange_rules().compute_fee(notional_filled, is_maker=is_maker, pair=self.pair)
        self.stats["fills"] = self.stats.get("fills", 0) + 1
        return {"filled_size": float(filled), "avg_fill_price": float(avg_price), "fee": float(fee)}
