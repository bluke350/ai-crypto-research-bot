from __future__ import annotations

import pandas as pd
import numpy as np

from src.execution.exchange_rules import get_exchange_rules
from src.execution.simulators import ExchangeSimulator
from src.execution.order_models import Order


def test_exchange_rules_and_simulator():
    rules = get_exchange_rules()
    xbt_rules = rules.get("XBT/USD")
    assert xbt_rules.lot_size > 0
    assert xbt_rules.min_notional > 0

    sim = ExchangeSimulator(pair="XBT/USD")
    # price 100, request size 0.00005 -> notional 0.005 < min_notional -> zero fill
    ord1 = Order(order_id="1", pair="XBT/USD", side="buy", size=0.00005)
    r1 = sim.place_order(ord1, market_price=100.0, is_maker=False)
    assert r1["filled_size"] == 0.0
    # request larger order that meets min_notional: size 0.2 -> notional 20 -> filled
    ord2 = Order(order_id="2", pair="XBT/USD", side="buy", size=0.2)
    r2 = sim.place_order(ord2, market_price=100.0, is_maker=False)
    assert r2["filled_size"] != 0.0
    assert r2["fee"] > 0.0
