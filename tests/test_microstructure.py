import math
from src.features.microstructure import (
    best_bid_ask,
    spread,
    bid_ask_imbalance,
    weighted_depth_imbalance,
    depth_imbalance_by_price_bands,
)


def make_orderbook(bids, asks):
    return {"bids": bids, "asks": asks}


def test_best_bid_ask_and_spread():
    ob = make_orderbook([(100.0, 1.0), (99.5, 2.0)], [(100.5, 1.5), (101.0, 1.0)])
    b, a = best_bid_ask(ob)
    assert b == 100.0
    assert a == 100.5
    s = spread(ob)
    assert math.isclose(s, 0.5)


def test_bid_ask_imbalance_top_n():
    ob = make_orderbook([(100.0, 5.0), (99.5, 2.0)], [(100.5, 3.0), (101.0, 1.0)])
    im = bid_ask_imbalance(ob, n=2)
    # B = 7, A = 4 -> (7-4)/(7+4) = 3/11
    assert math.isclose(im, 3.0/11.0, rel_tol=1e-6)


def test_weighted_depth_imbalance():
    ob = make_orderbook([(99.0, 10.0), (98.0, 5.0)], [(101.0, 8.0), (102.0, 2.0)])
    im = weighted_depth_imbalance(ob, n=2)
    # Weighted imbalance should be computable and finite
    assert isinstance(im, float)
    assert -1.0 <= im <= 1.0


def test_depth_imbalance_by_price_bands():
    ob = make_orderbook([(99.0, 10.0), (98.5, 5.0)], [(101.0, 8.0), (102.0, 2.0)])
    bands = [0.0, 0.01, 0.02]
    res = depth_imbalance_by_price_bands(ob, bands)
    # returns band keys
    assert isinstance(res, dict)
    # numeric values between -1 and 1
    for v in res.values():
        assert -1.0 <= v <= 1.0
