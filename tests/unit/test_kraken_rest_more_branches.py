import types
import pytest
import src.ingestion.providers.kraken_rest as kr


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


def noop_sleep(_=None):
    return None


def test_get_ohlc_error_then_success(monkeypatch):
    # First response contains an error (but attempts < max_attempts) then success
    err = {'error': ['E']}
    row = [1609459200, '30000.0', '30010.0', '29990.0', '30005.0', '30005.0', '1.5', '2']
    ok = {'result': {'XBT/USD': [row]}, 'error': []}
    monkeypatch.setattr(kr, 'requests', types.SimpleNamespace(get=make_get([err, ok])))
    monkeypatch.setattr(kr, 'cfg', {'retries': {'max_attempts': 3, 'backoff_ms': [1, 1, 1]}, 'rate_limits': {}})
    monkeypatch.setattr(kr, 'time', types.SimpleNamespace(sleep=noop_sleep))
    monkeypatch.setattr(kr, 'rl', types.SimpleNamespace(sleep_if_needed=lambda: None))

    df = kr.get_ohlc('XBT/USD', '1m', since=0)
    assert not df.empty


def test_get_ohlc_t_gt_end_break_raises(monkeypatch):
    # payload has an OHLC with t > end so loop breaks and df ends up empty -> ValueError
    row = [200, '1', '2', '3', '4', '0', '0', '0']
    resp1 = {'result': {'XBT/USD': [row]}, 'error': []}
    resp2 = {'result': {'XBT/USD': []}, 'error': []}
    monkeypatch.setattr(kr, 'requests', types.SimpleNamespace(get=make_get([resp1, resp2])))
    monkeypatch.setattr(kr, 'cfg', {'retries': {'max_attempts': 2}, 'rate_limits': {}})
    monkeypatch.setattr(kr, 'time', types.SimpleNamespace(sleep=noop_sleep))
    monkeypatch.setattr(kr, 'rl', types.SimpleNamespace(sleep_if_needed=lambda: None))

    with pytest.raises(ValueError):
        kr.get_ohlc('XBT/USD', '1m', since=0, end=100)


def test_get_ohlc_last_ts_ge_end_break_returns(monkeypatch):
    # payload with last_ts equal to end should break after appending rows and return df
    row = [300, '1', '2', '3', '4', '0', '0', '0']
    resp1 = {'result': {'XBT/USD': [row]}, 'error': []}
    resp2 = {'result': {'XBT/USD': []}, 'error': []}
    monkeypatch.setattr(kr, 'requests', types.SimpleNamespace(get=make_get([resp1, resp2])))
    monkeypatch.setattr(kr, 'cfg', {'retries': {'max_attempts': 2}, 'rate_limits': {}})
    monkeypatch.setattr(kr, 'time', types.SimpleNamespace(sleep=noop_sleep))
    monkeypatch.setattr(kr, 'rl', types.SimpleNamespace(sleep_if_needed=lambda: None))

    df = kr.get_ohlc('XBT/USD', '1m', since=0, end=300)
    assert not df.empty


def test_get_trades_last_first_and_payload(monkeypatch):
    # Ensure 'last' key appears first in result.items() iteration and is skipped
    trade = ['100.0', '0.1', '1609459200.5', 'b']
    # Put 'last' first in dict literal so iteration sees it first
    result = {'last': 'token', 'XBT/USD': [trade]}
    resp = {'result': result, 'error': []}
    monkeypatch.setattr(kr, 'requests', types.SimpleNamespace(get=make_get([resp, {'result': {'XBT/USD': []}, 'error': []}])))
    monkeypatch.setattr(kr, 'cfg', {'retries': {'max_attempts': 2}, 'rate_limits': {}})
    monkeypatch.setattr(kr, 'time', types.SimpleNamespace(sleep=noop_sleep))
    monkeypatch.setattr(kr, 'rl', types.SimpleNamespace(sleep_if_needed=lambda: None))

    df = kr.get_trades('XBT/USD', since=0)
    assert not df.empty


def test_get_trades_last_missing_raises(monkeypatch):
    # result contains no 'last' value -> should break and raise ValueError since no rows
    result = {'XBT/USD': [], 'last': None}
    resp = {'result': result, 'error': []}
    monkeypatch.setattr(kr, 'requests', types.SimpleNamespace(get=make_get([resp])))
    monkeypatch.setattr(kr, 'cfg', {'retries': {'max_attempts': 1}, 'rate_limits': {}})
    monkeypatch.setattr(kr, 'time', types.SimpleNamespace(sleep=noop_sleep))
    monkeypatch.setattr(kr, 'rl', types.SimpleNamespace(sleep_if_needed=lambda: None))

    with pytest.raises(ValueError):
        kr.get_trades('XBT/USD', since=0)


def test_get_trades_end_float_exception_handled(monkeypatch):
    # last cannot be parsed as float; exception should be caught and code should continue
    trade = ['100.0', '0.1', '1609459200.5', 'b']
    result = {'XBT/USD': [trade], 'last': 'not-a-number'}
    resp1 = {'result': result, 'error': []}
    resp2 = {'result': {'XBT/USD': []}, 'error': []}
    monkeypatch.setattr(kr, 'requests', types.SimpleNamespace(get=make_get([resp1, resp2])))
    monkeypatch.setattr(kr, 'cfg', {'retries': {'max_attempts': 2}, 'rate_limits': {}})
    monkeypatch.setattr(kr, 'time', types.SimpleNamespace(sleep=noop_sleep))
    monkeypatch.setattr(kr, 'rl', types.SimpleNamespace(sleep_if_needed=lambda: None))

    df = kr.get_trades('XBT/USD', since=0, end=1609459201)
    assert not df.empty
