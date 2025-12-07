from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PairRules:
    lot_size: float
    min_notional: float
    price_tick: Optional[float]
    maker_fee: float  # fraction, e.g., 0.0005 = 0.05%
    taker_fee: float


class ExchangeRules:
    """Registry of simple exchange rules (lot sizes, min notional, fees).

    This is a minimal, easily-extendable ruleset used by simulators and mappers.
    """

    def __init__(self):
        # sensible defaults and some sample pair overrides
        self.defaults = PairRules(lot_size=0.0001, min_notional=10.0, price_tick=None, maker_fee=0.0002, taker_fee=0.0007)
        self._pairs: Dict[str, PairRules] = {
            "XBT/USD": PairRules(lot_size=0.0001, min_notional=10.0, price_tick=0.01, maker_fee=0.0002, taker_fee=0.0007),
            "ETH/USD": PairRules(lot_size=0.001, min_notional=5.0, price_tick=0.01, maker_fee=0.0002, taker_fee=0.0007),
        }

    def get(self, pair: str) -> PairRules:
        return self._pairs.get(pair, self.defaults)

    def compute_fee(self, notional: float, *, is_maker: bool = False, pair: Optional[str] = None) -> float:
        rules = self.get(pair) if pair is not None else self.defaults
        rate = rules.maker_fee if is_maker else rules.taker_fee
        return float(abs(notional) * float(rate))


# singleton for convenience
_EXCHANGE_RULES = ExchangeRules()


def get_exchange_rules() -> ExchangeRules:
    return _EXCHANGE_RULES
