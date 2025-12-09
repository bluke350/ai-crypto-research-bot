import math

from src.execution.cost_models import FeeModel, SlippageModel, LatencySampler


def test_fee_model_fixed():
    fm = FeeModel(fixed_fee_pct=0.001)
    fee = fm.compute(1000.0)
    assert math.isclose(fee, 1.0, rel_tol=1e-9)


def test_fee_model_bps():
    fm = FeeModel(maker_bps=10, taker_bps=20)
    fee = fm.compute(500.0, is_maker=False)
    assert math.isclose(fee, 500.0 * 20 / 10000.0, rel_tol=1e-9)


def test_slippage_model_deterministic():
    sm = SlippageModel(k=0.1, daily_vol=0.02)
    pct = sm.estimate_pct(10000.0)
    assert pct >= 0.0


def test_slippage_model_stochastic_seeded():
    sm1 = SlippageModel(k=0.1, daily_vol=0.02, stochastic_sigma=0.5, seed=42)
    sm2 = SlippageModel(k=0.1, daily_vol=0.02, stochastic_sigma=0.5, seed=42)
    p1 = sm1.estimate_pct(10000.0)
    p2 = sm2.estimate_pct(10000.0)
    # seeded instances should produce same draw
    assert abs(p1 - p2) < 1e-12


def test_latency_sampler():
    ls = LatencySampler(base_ms=10, jitter_ms=5, seed=123)
    v = ls.sample_ms()
    assert v >= 10 and v <= 15
