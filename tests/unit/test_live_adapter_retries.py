from __future__ import annotations

from src.execution.order_models import Order
from src.execution.order_executor import LiveAdapterOrderExecutor


def test_live_adapter_retries_succeeds_after_retries():
    class FlakyClient:
        def __init__(self):
            self.calls = 0

        def place_order(self, order, market_price=None, is_maker=False):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("transient")
            return {'filled_size': order.size, 'avg_fill_price': market_price, 'fee': 0.0, 'order_id': 'ok'}

    client = FlakyClient()
    exec = LiveAdapterOrderExecutor(client=client, dry_run=False, max_retries=5, backoff_seconds=0.01)
    order = Order(order_id='r1', pair='XBT/USD', side='buy', size=0.1)
    res = exec.execute(order, market_price=1000.0)
    assert res['filled_size'] == 0.1
    assert exec.last_error is not None or exec.last_error == None
    assert any(f.get('order_id') == 'ok' for f in exec.fills)


def test_live_adapter_retries_all_fail_returns_dry():
    class AlwaysFailClient:
        def place_order(self, *args, **kwargs):
            raise RuntimeError("permanent")

    client = AlwaysFailClient()
    exec = LiveAdapterOrderExecutor(client=client, dry_run=False, max_retries=2, backoff_seconds=0.01)
    order = Order(order_id='r2', pair='XBT/USD', side='sell', size=0.2)
    res = exec.execute(order, market_price=2000.0)
    assert res['filled_size'] == 0.0
    assert 'error' in res and res['error'] is not None
    assert any(f.get('order_id') == 'r2' for f in exec.fills)
