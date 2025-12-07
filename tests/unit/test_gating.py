from __future__ import annotations

from src.execution.gating import RiskConfig, RiskGate
from src.execution.order_models import Order


def test_gate_blocks_large_order_notional():
    cfg = RiskConfig(max_order_notional=1000.0)
    gate = RiskGate(cfg)
    order = Order(order_id='o1', pair='XBT/USD', side='buy', size=1.0)
    ok, reason = gate.check_order(order, market_price=2000.0, positions={})
    assert not ok
    assert 'order_notional_too_large' in reason


def test_gate_allows_small_order_and_respects_position_cap():
    cfg = RiskConfig(max_order_notional=10000.0, max_position_abs=2.0)
    gate = RiskGate(cfg)
    order = Order(order_id='o2', pair='XBT/USD', side='buy', size=1.5)
    ok, reason = gate.check_order(order, market_price=1000.0, positions={'XBT/USD': 0.5})
    # projected position = 0.5 + 1.5 = 2.0 equals max -> allow
    assert ok

    # slightly larger order should be rejected
    large = Order(order_id='o3', pair='XBT/USD', side='buy', size=1.6)
    ok2, reason2 = gate.check_order(large, market_price=1000.0, positions={'XBT/USD': 0.5})
    assert not ok2
    assert 'position_limit_exceeded' in reason2


def test_daily_loss_trips_circuit():
    cfg = RiskConfig(daily_loss_limit=100.0)
    gate = RiskGate(cfg)
    order = Order(order_id='o4', pair='XBT/USD', side='sell', size=0.1)
    ok, reason = gate.check_order(order, market_price=1000.0, positions={}, realized_pnl=-150.0)
    assert not ok
    assert 'daily' in reason
