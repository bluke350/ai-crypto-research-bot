import json
import os
from pathlib import Path
import pandas as pd

import pytest

from orchestration import paper_live


class DummyWrapper:
    def __init__(self, target_series):
        self._series = target_series

    @classmethod
    def from_file(cls, path: str):
        raise RuntimeError("use patched from_file")

    def predicted_to_targets(self, prices: pd.DataFrame, method: str = "vol_norm", **kwargs):
        # return series aligned to prices.index[2:]
        idx = prices.index[2:]
        return pd.Series([self._series] * len(idx), index=idx, name="target")


@pytest.fixture(autouse=True)
def disable_health_server_env(monkeypatch):
    monkeypatch.setenv('WS_DISABLE_HEALTH_SERVER', '1')
    yield


def test_order_notional_rejection(tmp_path, monkeypatch):
    # patch ModelWrapper.from_file to return DummyWrapper that requests large targets
    def fake_from_file(path: str):
        return DummyWrapper(target_series=100.0)

    monkeypatch.setattr(paper_live, 'ModelWrapper', type('M', (), {'from_file': staticmethod(fake_from_file)}))

    out = paper_live.run_live(checkpoint_path='fake', prices_csv=None, out_root=str(tmp_path), max_ticks=5, max_order_notional=100.0)
    res = json.loads((Path(out) / 'result.json').read_text())
    # at least one execution should be present and rejected (filled_size == 0)
    executions = res.get('executions', [])
    assert len(executions) > 0
    assert any(float(exec_.get('filled_size', 0.0)) == 0.0 for exec_ in executions)


def test_max_position_circuit_breaker(tmp_path, monkeypatch):
    # patch ModelWrapper.from_file to return DummyWrapper that requests medium targets
    def fake_from_file(path: str):
        return DummyWrapper(target_series=1.0)

    monkeypatch.setattr(paper_live, 'ModelWrapper', type('M', (), {'from_file': staticmethod(fake_from_file)}))

    max_ticks = 50
    out = paper_live.run_live(checkpoint_path='fake', prices_csv=None, out_root=str(tmp_path), max_ticks=max_ticks, max_position=0.5)
    res = json.loads((Path(out) / 'result.json').read_text())
    pnl = res.get('pnl', [])
    # expect either the circuit breaker triggered (fewer ticks processed)
    # OR the risk gate prevented position growth (we'll see rejected executions recorded)
    if len(pnl) >= max_ticks:
        executions = json.loads((Path(out) / 'result.json').read_text()).get('executions', [])
        assert any(exec_.get('gate_allowed') is False for exec_ in executions), "expected gate rejections when circuit not tripped"