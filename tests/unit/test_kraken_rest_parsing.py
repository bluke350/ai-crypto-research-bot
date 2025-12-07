import types
import pandas as pd
import src.ingestion.providers.kraken_rest as kr


class DummyResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def make_get(responses):
    # closure that returns successive DummyResp objects
    calls = {'i': 0}

    def _get(*args, **kwargs):
        i = calls['i']
        calls['i'] += 1
        payload = responses[i] if i < len(responses) else responses[-1]
        return DummyResp(payload)

    return _get


def test_get_ohlc_parsing(monkeypatch):
    # prepare a single OHLC row and then an empty payload to stop the loop
    row = [1609459200, '30000.0', '30010.0', '29990.0', '30005.0', '30005.0', '1.5', '2']
    resp1 = {'result': {'XBT/USD': [row]}, 'error': []}
    resp2 = {'result': {'XBT/USD': []}, 'error': []}
    monkeypatch.setattr(kr, 'requests', types.SimpleNamespace(get=make_get([resp1, resp2])))
    # ensure cfg has small max_attempts
    monkeypatch.setattr(kr, 'cfg', {'retries': {'max_attempts': 2}, 'rate_limits': {}})

    df = kr.get_ohlc('XBT/USD', '1m', since=0)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'count']
    assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])


def test_get_trades_parsing(monkeypatch):
    # prepare a single trade and then result with no payload to stop
    trade = ['30005.0', '0.1', '1609459200.5', 'b']
    resp1 = {'result': {'XBT/USD': [trade], 'last': '1609459200.5'}, 'error': []}
    resp2 = {'result': {'XBT/USD': [], 'last': None}, 'error': []}
    monkeypatch.setattr(kr, 'requests', types.SimpleNamespace(get=make_get([resp1, resp2])))
    monkeypatch.setattr(kr, 'cfg', {'retries': {'max_attempts': 2}, 'rate_limits': {}})

    df = kr.get_trades('XBT/USD', since=0)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ['timestamp', 'price', 'size', 'side']
    assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
