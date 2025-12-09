from dataclasses import dataclass
from typing import Optional


@dataclass
class Order:
    order_id: str
    pair: str
    side: str  # 'buy' or 'sell'
    size: float
    price: Optional[float] = None
    type: str = "limit"  # 'market'|'limit'|'stop'


def validate_order(order: Order, rules: dict):
    min_size = rules.get("min_order_size", {}).get(order.pair)
    tick = rules.get("price_tick", {}).get(order.pair)
    if min_size is not None and abs(order.size) < min_size:
        raise ValueError(f"Order size {order.size} below min {min_size}")
    if order.price is not None and tick is not None:
        # price should be multiple of tick
        if round(order.price / tick) * tick != order.price:
            raise ValueError(f"Price {order.price} not aligned to tick {tick}")
