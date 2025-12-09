import pandas as pd
import numpy as np

from src.execution.execution import twap_schedule, vwap_allocation, simple_slippage


def test_twap_schedule_even():
    start = pd.Timestamp('2021-01-01T00:00:00Z')
    end = pd.Timestamp('2021-01-01T01:00:00Z')
    sched = twap_schedule(start, end, total_qty=100, intervals=4)
    assert len(sched) == 4
    # each allocation equal
    qtys = [q for _, q in sched]
    assert all(abs(q - 25.0) < 1e-8 for q in qtys)


def test_vwap_allocation_proportional():
    df = pd.DataFrame({
        'price': [1.0, 2.0, 3.0],
        'volume': [10.0, 20.0, 30.0]
    })
    out = vwap_allocation(df, total_qty=600)
    # allocations proportional to volume
    assert pytest_approx(out.loc[0, 'alloc_qty'], 100)
    assert pytest_approx(out.loc[1, 'alloc_qty'], 200)
    assert pytest_approx(out.loc[2, 'alloc_qty'], 300)


def pytest_approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_simple_slippage():
    s = simple_slippage(1000, 1.0, impact_per_unit=1e-5)
    assert abs(s - 0.01) < 1e-12
