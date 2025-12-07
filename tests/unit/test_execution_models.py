import math
from src.execution.fee_models import compute_fee
from src.execution.slippage import sqrt_impact_slippage
from src.execution.latency import LatencyModel
from src.execution.simulator import Simulator
from src.execution.order_models import Order


def test_fee_compute():
    fee = compute_fee(1000.0, 'buy', is_maker=False, maker_bps=10, taker_bps=20)
    assert math.isclose(fee, 1000.0 * 0.002)


def test_slippage():
    s = sqrt_impact_slippage(1000.0, daily_vol=0.02, k=0.1)
    assert s > 0


def test_latency_model():
    lm = LatencyModel(base_ms=10, jitter_ms=5, seed=42)
    v = lm.sample()
    assert v >= 10


def test_simulator_fill():
    sim = Simulator(rules={"maker_bps": 10, "taker_bps": 20}, latency_model=LatencyModel(seed=1))
    order = Order(order_id="o1", pair="XBT/USD", side="buy", size=0.1, price=50000.0)
    fill = sim.place_order(order, market_price=50000.0, is_maker=False)
    assert fill["status"] == "filled"
    assert "fee" in fill and "slippage" in fill
