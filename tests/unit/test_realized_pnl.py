from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from orchestration import paper_live


class SequenceModelWrapper:
    def __init__(self, targets):
        self.targets = list(targets)

    @classmethod
    def from_file(cls, path: str):
        raise RuntimeError("use patched from_file")

    def predicted_to_targets(self, prices: pd.DataFrame, method: str = "vol_norm", **kwargs):
        idx = prices.index[2:]
        # ensure length matches
        out = self.targets[: len(idx)] + [self.targets[-1]] * max(0, len(idx) - len(self.targets))
        return pd.Series(out, index=idx, name='target')


def _make_prices_csv(tmp_path, prices):
    df = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=len(prices), freq='T'),
        'close': prices,
    })
    p = tmp_path / 'prices.csv'
    df.to_csv(p, index=False)
    return str(p)


def test_realized_pnl_partial_close(tmp_path, monkeypatch):
    # Prices: index 0..5 => model starts at index 2 (price 102)
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    csv = _make_prices_csv(tmp_path, prices)

    # targets: first target opens 1.0, next target 0.0 causes smoothed to 0.3 -> sell 0.7
    targets = [1.0, 0.0, 0.0, 0.0]
    monkeypatch.setattr(paper_live, 'ModelWrapper', type('M', (), {'from_file': staticmethod(lambda p: SequenceModelWrapper(targets))}))

    # monkeypatch simulator to always full-fill at market price with zero fee
    from src.execution.simulators import ExchangeSimulator

    monkeypatch.setattr(ExchangeSimulator, 'place_order', lambda self, order, market_price, is_maker=False: {'filled_size': float(order.size), 'avg_fill_price': float(market_price), 'fee': 0.0})

    out = paper_live.run_live(checkpoint_path='fake', prices_csv=csv, out_root=str(tmp_path), max_ticks=6)
    gate_state = json.loads((Path(out) / 'gate_state.json').read_text())
    # recompute realized pnl from execution records to validate consistency
    res = json.loads((Path(out) / 'result.json').read_text())
    # only consider per-tick execution rows (have timestamp) and were allowed by gate
    execs = [e for e in res.get('executions', []) if 'timestamp' in e and float(e.get('filled_size', 0.0)) != 0.0 and e.get('gate_allowed')]
    pos = 0.0
    avg_entry = 0.0
    realized = 0.0
    for e in execs:
        filled = float(e.get('filled_size', 0.0))
        price = float(e.get('avg_fill_price'))
        prev = pos
        new_pos = prev + filled
        if prev != 0 and (prev * filled) < 0:
            closed = min(abs(filled), abs(prev))
            if prev > 0:
                realized += closed * (price - avg_entry)
            else:
                realized += closed * (avg_entry - price)
        # update avg entry
        if new_pos == 0:
            avg_entry = 0.0
        else:
            if prev == 0 or (prev * filled) > 0:
                total_qty = abs(prev) + abs(filled)
                if total_qty > 0:
                    avg_entry = ((abs(prev) * avg_entry) + (abs(filled) * price)) / total_qty
            else:
                if abs(new_pos) > 0 and (prev * new_pos) < 0:
                    avg_entry = price
        pos = new_pos

    assert abs(realized - float(gate_state.get('realized_pnl', 0.0))) < 1e-6


def test_realized_pnl_full_close_no_smoothing(tmp_path, monkeypatch):
    prices = [100.0, 101.0, 110.0, 115.0]
    csv = _make_prices_csv(tmp_path, prices)

    # with target_smoothing_alpha=0.0 smoothed == target; open 1.0, then 0.0 fully closes
    targets = [1.0, 0.0]
    monkeypatch.setattr(paper_live, 'ModelWrapper', type('M', (), {'from_file': staticmethod(lambda p: SequenceModelWrapper(targets))}))

    from src.execution.simulators import ExchangeSimulator
    monkeypatch.setattr(ExchangeSimulator, 'place_order', lambda self, order, market_price, is_maker=False: {'filled_size': float(order.size), 'avg_fill_price': float(market_price), 'fee': 0.0})

    out = paper_live.run_live(checkpoint_path='fake', prices_csv=csv, out_root=str(tmp_path), max_ticks=4, target_smoothing_alpha=0.0)
    gate_state = json.loads((Path(out) / 'gate_state.json').read_text())
    # open at price index 2 (110.0) and close at index 3 (115.0) -> pnl = 1 * (115 - 110) = 5
    assert abs(gate_state.get('realized_pnl', 0.0) - 5.0) < 1e-6


def test_realized_pnl_flip_no_smoothing(tmp_path, monkeypatch):
    prices = [100.0, 100.0, 100.0, 105.0]
    csv = _make_prices_csv(tmp_path, prices)

    # alpha=0.0, targets: open 1.0 then -2.0 -> flip to -2.0, closed =1.0 at sell price (105)
    targets = [1.0, -2.0]
    monkeypatch.setattr(paper_live, 'ModelWrapper', type('M', (), {'from_file': staticmethod(lambda p: SequenceModelWrapper(targets))}))

    from src.execution.simulators import ExchangeSimulator
    monkeypatch.setattr(ExchangeSimulator, 'place_order', lambda self, order, market_price, is_maker=False: {'filled_size': float(order.size), 'avg_fill_price': float(market_price), 'fee': 0.0})

    out = paper_live.run_live(checkpoint_path='fake', prices_csv=csv, out_root=str(tmp_path), max_ticks=4, target_smoothing_alpha=0.0)
    gate_state = json.loads((Path(out) / 'gate_state.json').read_text())
    # realized from closing original long of 1.0 at price index 3 = 105 - entry 100 = 5
    assert abs(gate_state.get('realized_pnl', 0.0) - 5.0) < 1e-6
