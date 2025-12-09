import pandas as pd

from src.validation.walk_forward import evaluate_walk_forward


class DummySizer:
    def __init__(self):
        self.calls = []

    def observe(self, price):
        # record observation for diagnostics
        self.calls.append(("observe", price))

    def size(self, target_signal, price, equity, current_position):
        self.calls.append(("size", target_signal, price, equity, current_position))
        # return target as units to keep behavior simple
        return target_signal


class DummySim:
    def __init__(self):
        self.stats = {}

    def place_order(self, order, market_price=None, is_maker=False):
        return {
            "order_id": order.order_id,
            "filled_size": order.size,
            "avg_fill_price": market_price,
            "notional": abs(order.size * (market_price or 0)),
            "fee": 0.0,
            "slippage": 0.0,
            "latency_ms": 0.0,
            "status": "filled",
        }


def test_walk_forward_passes_sizer_through_backtest():
    timestamps = pd.date_range("2021-01-01", periods=6, freq="T", tz="UTC")
    prices = pd.DataFrame({"timestamp": timestamps, "close": [100, 101, 102, 103, 104, 105]})

    class Strat:
        def __init__(self, **_):
            pass

        def generate_targets(self, df):
            return pd.Series(1.0, index=df.index)

    sizer = DummySizer()
    sim_factory = lambda: DummySim()

    out = evaluate_walk_forward(
        prices=prices,
        targets=None,
        simulator=sim_factory,
        window=3,
        step=2,
        tuner=None,
        param_space=None,
        strategy_factory=Strat,
        sizer=sizer,
    )

    # sizer should have been used for both observe/size calls
    observed = [c for c in sizer.calls if c[0] == "size"]
    assert observed, "sizer.size should be called during evaluation"
    # ensure at least one fold evaluated
    assert out["folds"], "walk-forward should return folds"
