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


def test_get_ohlc_raises_on_kraken_error(monkeypatch, tmp_path):
    # Force max_attempts to 1 for a quick test
    monkeypatch.setattr(kr, 'cfg', {'retries': {'max_attempts': 1}, 'rate_limits': {}})

    def fake_get(*args, **kwargs):
        return DummyResp({'error': ['SOME_ERROR']})

    monkeypatch.setattr(kr, 'requests', types.SimpleNamespace(get=fake_get))

    with pytest.raises(RuntimeError):
        kr.get_ohlc('XBT/USD', '1m', since=0)
