import pandas as pd
import pytest

from src.execution.position_sizer import VolatilityRiskSizer
from src.execution.simulator import Simulator
from src.validation.backtester import run_backtest


def test_volatility_risk_sizer_caps_by_leverage():
    sizer = VolatilityRiskSizer(
        risk_fraction=0.01,
        vol_lookback=5,
        vol_floor=0.001,
        stop_multiple=1.0,
        max_leverage=1.0,
        max_position_fraction=1.0,
        lot_size=1.0,
    )
    # observe flat prices to exercise vol_floor path
    for p in [100.0, 100.0, 100.0]:
        sizer.observe(p)

    size = sizer.size(target_signal=1.0, price=100.0, equity=10_000.0, current_position=0.0)
    # risk=100, stop_dist=0.1 -> raw size=1000 units, capped by 1x leverage (10k notional)
    assert size == pytest.approx(100.0)


def test_backtest_uses_sizer_for_auto_sizing():
    timestamps = pd.date_range("2021-01-01", periods=3, freq="T", tz="UTC")
    prices = pd.DataFrame({"timestamp": timestamps, "close": [100.0, 102.0, 101.0]})
    targets = pd.Series([1.0, 1.0, 0.0])

    sizer = VolatilityRiskSizer(
        risk_fraction=0.02,
        vol_lookback=2,
        vol_floor=0.001,
        stop_multiple=1.0,
        max_leverage=1.0,
        max_position_fraction=1.0,
        lot_size=1.0,
    )
    try:
        from src.execution.cost_models import FeeModel, SlippageModel, LatencySampler
    except Exception:
        FeeModel = None
        SlippageModel = None
        LatencySampler = None
    fee = FeeModel(fixed_fee_pct=0.0) if FeeModel is not None else None
    slip = SlippageModel(fixed_slippage_pct=0.0) if SlippageModel is not None else None
    lat = LatencySampler(base_ms=0, jitter_ms=0) if LatencySampler is not None else None
    sim = Simulator(fee_model=fee, slippage_model=slip, latency_model=lat)

    out = run_backtest(prices, targets, sim, initial_cash=10_000.0, sizer=sizer)

    execs = out["executions"]
    assert len(execs) == 2  # enter then exit when target goes to zero
    first_size = execs.iloc[0]["filled_size"]
    exit_size = execs.iloc[1]["filled_size"]

    # entry size should respect leverage cap and be positive
    assert first_size > 0
    assert first_size <= 100.0
    # exit should roughly flatten the position
    assert exit_size < 0
    assert pytest.approx(first_size + exit_size, rel=1e-3, abs=1e-3) == 0
    # equity series should exist
    assert not out["pnl"].empty
