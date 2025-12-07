from __future__ import annotations

import importlib
from typing import Any, Dict

from src.execution.order_models import Order


class ExchangeClientWrapper:
    """Normalize `place_order` across different exchange client libraries.

    Supported providers: `ccxt` (Kraken via ccxt.kraken), `krakenex`.
    If a real client instance is passed, the wrapper will attempt to adapt to
    common methods (`create_order`, `place_order`, `submit_order`, `query`).
    """

    def __init__(self, client: Any):
        self.client = client

    @classmethod
    def from_provider(cls, provider: str, api_key: str, api_secret: str) -> "ExchangeClientWrapper":
        p = (provider or '').lower()
        if p in ('ccxt', 'kraken'):
            ccxt = importlib.import_module('ccxt')
            ex = ccxt.kraken({'apiKey': api_key, 'secret': api_secret})
            return cls(ex)
        if p == 'krakenex':
            krakenex = importlib.import_module('krakenex')
            api = krakenex.API(api_key, api_secret)
            return cls(api)
        raise ValueError(f"unsupported provider: {provider}")

    def place_order(self, order: Order, market_price: float | None = None, is_maker: bool = False) -> Dict[str, Any]:
        # Try ccxt-style client
        if hasattr(self.client, 'create_order'):
            # ccxt: create_order(symbol, type, side, amount, price=None, params={})
            symbol = order.pair
            otype = 'market' if order.type == 'market' else 'limit'
            price = None if order.type == 'market' else (order.price or market_price)
            res = self.client.create_order(symbol, otype, order.side, abs(order.size), price)
            # Normalize response
            return {
                'filled_size': float(res.get('filled', res.get('amount_filled', 0) or 0)),
                'avg_fill_price': float(res.get('average', res.get('price') or market_price or 0.0)),
                'fee': float(res.get('fee', 0) or 0),
                'order_id': str(res.get('id', '')),
                'pair': order.pair,
                'side': order.side,
                'requested_size': order.size,
            }

        # Try krakenex-style client (API.query)
        if hasattr(self.client, 'query_private') or hasattr(self.client, 'query'):
            # Build params for Kraken AddOrder
            params = {
                'pair': order.pair,
                'type': order.side,
                'ordertype': 'market' if order.type == 'market' else 'limit',
                'volume': abs(order.size),
            }
            if order.price is not None and order.type != 'market':
                params['price'] = order.price
            # Prefer query_private if available
            query = getattr(self.client, 'query_private', None) or getattr(self.client, 'query')
            res = query('AddOrder', params)
            # The krakenex response shape varies; attempt to parse
            txid = ''
            filled = 0.0
            avg = market_price or 0.0
            fee = 0.0
            try:
                txid = ','.join(res.get('result', {}).get('txid', []))
            except Exception:
                txid = ''
            return {
                'filled_size': float(filled),
                'avg_fill_price': float(avg),
                'fee': float(fee),
                'order_id': txid,
                'pair': order.pair,
                'side': order.side,
                'requested_size': order.size,
            }

        # Try generic place_order/submit_order
        if hasattr(self.client, 'place_order'):
            return self.client.place_order(order, market_price=market_price, is_maker=is_maker)
        if hasattr(self.client, 'submit_order'):
            return self.client.submit_order(order, price=market_price)

        raise NotImplementedError('client adapter does not support placing orders')
