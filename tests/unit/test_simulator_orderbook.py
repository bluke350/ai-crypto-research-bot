from __future__ import annotations

from src.execution.simulators import ExchangeSimulator
from src.execution.order_models import Order


def test_orderbook_full_fill_buy():
    # asks: price asc
    ob = {"asks": [(100.0, 1.0), (101.0, 2.0)], "bids": [(99.0, 1.0)]}
    sim = ExchangeSimulator(pair='XBT/USD', fill_model='orderbook', orderbook=ob)
    order = Order(order_id='o1', pair='XBT/USD', side='buy', size=1.0, type='limit', price=101.0)
    res = sim.place_order(order, market_price=100.5)
    assert abs(res['filled_size'] - 1.0) < 1e-3
    # average should be <= 101 and >=100
    assert 100.0 <= res['avg_fill_price'] <= 101.0


def test_orderbook_partial_fill_due_to_depth():
    ob = {"asks": [(100.0, 0.3), (101.0, 0.1)], "bids": [(99.0, 1.0)]}
    sim = ExchangeSimulator(pair='XBT/USD', fill_model='orderbook', orderbook=ob)
    order = Order(order_id='o2', pair='XBT/USD', side='buy', size=1.0, type='limit', price=101.0)
    res = sim.place_order(order, market_price=100.5)
    # available depth = 0.4 (allow small rounding tolerance)
    assert abs(res['filled_size'] - 0.4) < 1e-3
    assert 100.0 <= res['avg_fill_price'] <= 101.0
