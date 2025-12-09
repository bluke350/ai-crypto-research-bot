from src.execution.order_models import Order, validate_order


def test_validate_order_raises_min_size():
    rules = {"min_order_size": {"XBT/USD": 1.0}}
    order = Order(order_id="1", pair="XBT/USD", side="buy", size=0.5, price=100.0)

    try:
        validate_order(order, rules)
        raise AssertionError("validate_order did not raise for small size")
    except ValueError as e:
        assert "below min" in str(e)


def test_validate_order_price_tick_misaligned():
    rules = {"price_tick": {"XBT/USD": 0.5}}
    # price 0.75 is not a multiple of 0.5
    order = Order(order_id="2", pair="XBT/USD", side="sell", size=2.0, price=0.75)

    try:
        validate_order(order, rules)
        raise AssertionError("validate_order did not raise for misaligned tick")
    except ValueError as e:
        assert "not aligned to tick" in str(e)


def test_validate_order_passes_for_valid_order():
    rules = {"min_order_size": {"XBT/USD": 0.1}, "price_tick": {"XBT/USD": 0.25}}
    # price 1.25 is multiple of 0.25
    order = Order(order_id="3", pair="XBT/USD", side="buy", size=1.0, price=1.25)

    # Should not raise
    validate_order(order, rules)
