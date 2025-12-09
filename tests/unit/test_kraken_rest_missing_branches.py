import builtins
import importlib
import sys
from collections import OrderedDict
import types
import pytest

import src.ingestion.providers as providers


def test_dummy_requests_fallback(monkeypatch):
    # Reload the module while making 'requests' import fail to exercise the fallback
    mod_name = 'src.ingestion.providers.kraken_rest'
    orig = sys.modules.get(mod_name)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'requests':
            raise ImportError('no requests')
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    try:
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        mod = importlib.import_module(mod_name)
        # module should expose _DummyRequests and requests should be instance
        assert hasattr(mod, '_DummyRequests')
        assert isinstance(mod.requests, mod._DummyRequests)
    finally:
        # restore
        monkeypatch.setattr(builtins, '__import__', real_import)
        if orig is not None:
            sys.modules[mod_name] = orig


class DummyResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def make_get(responses):
    calls = {'i': 0}

    def _get(*args, **kwargs):
        i = calls['i']
        calls['i'] += 1
        payload = responses[i] if i < len(responses) else responses[-1]
        return DummyResp(payload)

    return _get


def test_get_ohlc_retry_then_success(monkeypatch):
    kr = providers.kraken_rest
    # First response has error, second returns payload
    resp_err = {'error': ['E'], 'result': {}}
    row = [1609459200, '30000', '30010', '29990', '30005', '30005', '1.5', '2']
    resp_ok = {'error': [], 'result': {'XBT/USD': [row]}}
    monkeypatch.setattr(kr, 'requests', types.SimpleNamespace(get=make_get([resp_err, resp_ok])))
    monkeypatch.setattr(kr, 'cfg', {'retries': {'max_attempts': 2, 'backoff_ms': [10] }, 'rate_limits': {}})
    monkeypatch.setattr('time.sleep', lambda *_: None)

    df = kr.get_ohlc('XBT/USD', '1m', since=0)
    assert not df.empty


def test_get_ohlc_for_loop_break_on_t_greater_than_end(monkeypatch):
    kr = providers.kraken_rest
    # one ohlc with t > end should trigger inner break
    row = [200, '1', '2', '3', '4', '0', '0', '0']
    resp1 = {'error': [], 'result': {'XBT/USD': [row]}}
    resp2 = {'error': [], 'result': {'XBT/USD': []}}
    monkeypatch.setattr(kr, 'requests', types.SimpleNamespace(get=make_get([resp1, resp2])))
    monkeypatch.setattr(kr, 'cfg', {'retries': {'max_attempts': 2}, 'rate_limits': {}})
    monkeypatch.setattr('time.sleep', lambda *_: None)

    # end less than row[0] so break at line for t > end; rows empty -> ValueError
    with pytest.raises(ValueError):
        kr.get_ohlc('XBT/USD', '1m', since=0, end=100)


def test_get_ohlc_last_ts_break_after_loop(monkeypatch):
    kr = providers.kraken_rest
    # two rows, last_ts equals end, so after loop last_ts >= end triggers break
    row1 = [100, '1', '2', '3', '4', '0', '0', '0']
    row2 = [200, '1', '2', '3', '4', '0', '0', '0']
    resp1 = {'error': [], 'result': {'XBT/USD': [row1, row2]}}
    resp2 = {'error': [], 'result': {'XBT/USD': []}}
    monkeypatch.setattr(kr, 'requests', types.SimpleNamespace(get=make_get([resp1, resp2])))
    monkeypatch.setattr(kr, 'cfg', {'retries': {'max_attempts': 2}, 'rate_limits': {}})
    monkeypatch.setattr('time.sleep', lambda *_: None)

    df = kr.get_ohlc('XBT/USD', '1m', since=0, end=200)
    assert not df.empty


def test_get_trades_continue_on_last_key(monkeypatch):
    kr = providers.kraken_rest
    trade = ['30005.0', '0.1', '1609459200.5', 'b']
    # Put 'last' first to trigger the continue branch when iterating result.items()
    result = OrderedDict([('last', 'token'), ('XBT/USD', [trade])])
    resp = {'error': [], 'result': result}
    resp2 = {'error': [], 'result': {'XBT/USD': []}}
    monkeypatch.setattr(kr, 'requests', types.SimpleNamespace(get=make_get([resp, resp2])))
    monkeypatch.setattr(kr, 'cfg', {'retries': {'max_attempts': 2}, 'rate_limits': {}})
    monkeypatch.setattr('time.sleep', lambda *_: None)

    df = kr.get_trades('XBT/USD', since=0)
    assert not df.empty


def test_get_trades_break_when_no_last(monkeypatch):
    kr = providers.kraken_rest
    trade = ['30005.0', '0.1', '1609459200.5', 'b']
    # result has payload and last is falsy -> last = None -> triggers break at line 128
    result = OrderedDict([('XBT/USD', [trade]), ('last', None)])
    resp = {'error': [], 'result': result}
    resp2 = {'error': [], 'result': {'XBT/USD': []}}
    monkeypatch.setattr(kr, 'requests', types.SimpleNamespace(get=make_get([resp, resp2])))
    monkeypatch.setattr(kr, 'cfg', {'retries': {'max_attempts': 2}, 'rate_limits': {}})
    monkeypatch.setattr('time.sleep', lambda *_: None)

    df = kr.get_trades('XBT/USD', since=0)
    assert not df.empty


def test_get_trades_end_float_and_exception_branches(monkeypatch):
    kr = providers.kraken_rest
    trade = ['30005.0', '0.1', '1609459200.5', 'b']
    # Case 1: last numeric >= end -> break
    result1 = {'XBT/USD': [trade], 'last': '200'}
    # Case 2: last non-numeric -> float() raises and except branch executed
    result2 = {'XBT/USD': [trade], 'last': 'notnum'}

    # make result2 the second response so the second get_trades call sees it
    monkeypatch.setattr(kr, 'requests', types.SimpleNamespace(get=make_get([
        {'error': [], 'result': result1},
        {'error': [], 'result': result2},
        {'error': [], 'result': {'XBT/USD': [], 'last': None}},
        {'error': [], 'result': {'XBT/USD': [], 'last': None}},
    ])))
    monkeypatch.setattr(kr, 'cfg', {'retries': {'max_attempts': 2}, 'rate_limits': {}})
    monkeypatch.setattr('time.sleep', lambda *_: None)

    # first call: end check should break and return df
    df1 = kr.get_trades('XBT/USD', since=0, end=100)
    assert not df1.empty

    # second scenario: last non-numeric should not raise
    df2 = kr.get_trades('XBT/USD', since=0, end=100)
    # since we provided payload, df2 should not be empty (function handles exception silently)
    assert not df2.empty


def test_get_trades_empty_raises(monkeypatch):
    kr = providers.kraken_rest
    # responses produce no payload -> should raise ValueError
    monkeypatch.setattr(kr, 'requests', types.SimpleNamespace(get=make_get([
        {'error': [], 'result': {'XBT/USD': []}},
    ])))
    monkeypatch.setattr(kr, 'cfg', {'retries': {'max_attempts': 1}, 'rate_limits': {}})
    monkeypatch.setattr('time.sleep', lambda *_: None)

    with pytest.raises(ValueError):
        kr.get_trades('XBT/USD', since=0)
