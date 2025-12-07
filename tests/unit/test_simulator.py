import pytest
from src.execution.simulator import Simulator


def test_partial_fill_slices_aggregation():
    sim = Simulator(rules={"partial_fill_fraction": 1.0, "partial_fill_slices": 4})
    from src.execution.order_models import Order
    order = Order(order_id="o1", pair="XBT/USD", side="buy", size=1.0)
    fill = sim.place_order(order, market_price=100.0)
    assert fill["filled_size"] == 1.0
    assert "slices" in fill
    assert len(fill["slices"]) == 4


def test_book_depth_reduces_slippage():
    # price and notional same, different book_depth should reduce slippage
    order = None
    from src.execution.order_models import Order
    order = Order(order_id="o2", pair="XBT/USD", side="buy", size=1.0)
    sim_shallow = Simulator(rules={"partial_fill_fraction": 1.0, "book_depth": 1.0, "slippage_k": 0.1})
    sim_deep = Simulator(rules={"partial_fill_fraction": 1.0, "book_depth": 16.0, "slippage_k": 0.1})
    f1 = sim_shallow.place_order(order, market_price=100.0)
    f2 = sim_deep.place_order(order, market_price=100.0)
    assert abs(f2["slippage"]) < abs(f1["slippage"]) or f2["slippage"] == 0
