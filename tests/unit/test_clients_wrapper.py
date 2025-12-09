from __future__ import annotations

from src.execution.clients import ExchangeClientWrapper
from src.execution.order_models import Order


def test_ccxt_wrapper_calls_create_order():
    class FakeCCXT:
        def __init__(self):
            self.calls = []

        def create_order(self, symbol, otype, side, amount, price=None):
            self.calls.append((symbol, otype, side, amount, price))
            return {'filled': amount, 'average': price or 123.45, 'id': 'ord123', 'fee': 0.01}

    fake = FakeCCXT()
    wrapper = ExchangeClientWrapper(fake)
    order = Order(order_id='o1', pair='XBT/USD', side='buy', size=0.5, price=200.0, type='limit')
    res = wrapper.place_order(order, market_price=199.0)
    assert res['order_id'] == 'ord123'
    assert res['filled_size'] == 0.5
    assert res['avg_fill_price'] == 200.0
    assert res['pair'] == 'XBT/USD'


def test_krakenex_wrapper_calls_query_private():
    class FakeKrakenEx:
        def query_private(self, method, params):
            assert method == 'AddOrder'
            # return a structure similar to krakenex
            return {'result': {'txid': ['TX1']}}

    fake = FakeKrakenEx()
    wrapper = ExchangeClientWrapper(fake)
    order = Order(order_id='o2', pair='XBT/USD', side='sell', size=1.0, type='market')
    res = wrapper.place_order(order, market_price=321.0)
    assert res['order_id'] == 'TX1' or res['order_id'] == 'TX1'
    assert res['requested_size'] == 1.0
