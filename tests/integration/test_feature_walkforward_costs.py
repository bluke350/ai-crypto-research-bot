import os
import pandas as pd

from src.utils.io import load_prices_csv
from src.execution.cost_models import FeeModel, SlippageModel, LatencySampler
from src.execution.simulator import Simulator
from src.validation.walk_forward import evaluate_walk_forward


def test_walk_forward_with_cost_models(tmp_path):
    # use sample CSV from repo
    csv = os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'sample_prices_for_cli.csv')
    csv = os.path.abspath(csv)
    prices = load_prices_csv(csv, dedupe='first')
    # ensure timestamp column present and sorted
    if 'timestamp' in prices.columns:
        prices['timestamp'] = pd.to_datetime(prices['timestamp'], utc=True)
    # simple zero targets aligned to prices
    targets = pd.Series(0.0, index=prices.index)

    # build cost models
    fee = FeeModel(fixed_fee_pct=0.0005)
    slip = SlippageModel(fixed_slippage_pct=0.001)
    lat = LatencySampler(base_ms=5, jitter_ms=2, seed=0)

    # simulator factory
    sim_factory = lambda: Simulator(fee_model=fee, slippage_model=slip, latency_model=lat, seed=0)

    # run short walk-forward
    res = evaluate_walk_forward(prices=prices, targets=targets, simulator=sim_factory, window=30, step=10)
    assert 'folds' in res
    assert isinstance(res['folds'], list)
    # check metrics presence
    for f in res['folds']:
        assert 'metrics' in f and 'final_value' in f['metrics']