import os
import sys
import types

import pytest


def test_legacy_env_vars_set_compression_days(monkeypatch):
    # set legacy hours-based env var to ensure conversion to days occurs
    monkeypatch.setenv("WS_WAL_ARCHIVE_COMPRESS_AFTER_HOURS", "48")
    monkeypatch.setenv("WS_WAL_ARCHIVE_INTERVAL_HOURS", "12")

    # import here so module reads current environment
    from src.ingestion.providers.kraken_ws import KrakenWSClient

    client = KrakenWSClient(out_root='.')

    # 48 hours -> 2 days
    assert abs(client._wal_compress_after_days - 2.0) < 1e-6
    assert abs(client._wal_compress_interval_hours - 12.0) < 1e-6


def test_prometheus_metrics_initialization(monkeypatch):
    # create a fake prometheus_client module with required symbols
    fake = types.ModuleType("prometheus_client")

    called = {}

    def fake_start_http_server(port):
        called['started'] = int(port)

    class FakeCounter:
        def __init__(self, *args, **kwargs):
            called['counter'] = True

    class FakeGauge:
        def __init__(self, *args, **kwargs):
            called['gauge'] = True
            # provide a simple _value object with .get() used in health handler
            self._value = types.SimpleNamespace(get=lambda: 123)

    fake.start_http_server = fake_start_http_server
    fake.Counter = FakeCounter
    fake.Gauge = FakeGauge

    # insert fake module into sys.modules so import in the client picks it up
    monkeypatch.setitem(sys.modules, 'prometheus_client', fake)

    # ensure environment uses the default health port or a small one
    monkeypatch.setenv('WS_METRICS_PORT', '9009')

    from src.ingestion.providers.kraken_ws import KrakenWSClient

    client = KrakenWSClient(out_root='.')

    # metrics objects should have been created
    assert getattr(client, 'connect_attempts', None) is not None
    assert getattr(client, 'last_processed_ts', None) is not None
    assert getattr(client, 'wal_queue_length', None) is not None
    # start_http_server should have been called with our port
    assert called.get('started') == 9009
