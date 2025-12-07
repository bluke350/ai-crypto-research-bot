import sys
import types
from pathlib import Path
import json

import pytest

import importlib
import orchestration


def test_prometheus_integration_monkeypatched(tmp_path, monkeypatch):
    # create a fake prometheus_client module and insert into sys.modules
    mod = types.ModuleType('prometheus_client')
    def start_http_server(port):
        # record port for introspection
        setattr(mod, '_started_port', port)
    class DummyCounter:
        def __init__(self, *args, **kwargs):
            pass
        def inc(self):
            pass
    class DummyGauge:
        def __init__(self, *args, **kwargs):
            pass
        def set(self, v):
            pass
    mod.start_http_server = start_http_server
    mod.Counter = DummyCounter
    mod.Gauge = DummyGauge
    monkeypatch.setitem(sys.modules, 'prometheus_client', mod)

    # reload orchestration.paper_live so it re-evaluates PROM_AVAILABLE with our fake module
    paper_live = importlib.import_module('orchestration.paper_live')
    importlib.reload(paper_live)

    # patch ModelWrapper.from_file to avoid needing a real checkpoint file
    class DummyWrapper:
        @classmethod
        def from_file(cls, path: str):
            return cls()

        def predicted_to_targets(self, prices, method='vol_norm', **kwargs):
            idx = prices.index[2:]
            import pandas as pd

            return pd.Series([0.0] * len(idx), index=idx, name='target')

    monkeypatch.setattr(paper_live, 'ModelWrapper', DummyWrapper)

    out = paper_live.run_live(checkpoint_path='fake_ckpt', prices_csv=None, out_root=str(tmp_path), max_ticks=3, prom_port=9005)
    # verify result.json created and run_plan exists
    result_path = Path(out) / 'result.json'
    run_plan_path = Path(out) / 'run_plan.json'
    assert result_path.exists()
    assert run_plan_path.exists()
    # prometheus fake should have recorded started port
    assert getattr(sys.modules['prometheus_client'], '_started_port', None) == 9005