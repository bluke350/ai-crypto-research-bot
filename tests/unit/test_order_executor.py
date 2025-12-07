from __future__ import annotations

import pytest

from src.execution.order_executor import PaperOrderExecutor
from src.execution.order_models import Order


def test_paper_executor_fill():
    sim = PaperOrderExecutor()
    ord = Order(order_id='t1', pair='XBT/USD', side='buy', size=1.0, price=None)
    fill = sim.execute(ord, market_price=100.0)
    assert 'filled_size' in fill
    assert 'avg_fill_price' in fill
    assert 'fee' in fill
    # record kept
    assert len(sim.fills) == 1
