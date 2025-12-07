import asyncio
import pytest
from unittest import mock

from src.ingestion.providers.kraken_ws import KrakenWSClient


class DummyWS:
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        return False
    def __aiter__(self):
        return self
    async def __anext__(self):
        raise StopAsyncIteration
    async def close(self):
        pass


@pytest.mark.asyncio
async def test_run_connect_attempts_full_jitter(monkeypatch):
    client = KrakenWSClient(out_root='.')
    client._backoff_test_mode = True
    client.backoff_policy = 'full-jitter'

    calls = {'i': 0}

    async def fake_connect(*args, **kwargs):
        i = calls['i']
        calls['i'] += 1
        if i < 2:
            raise Exception('connect fail')
        return DummyWS()

    import websockets
    monkeypatch.setattr(websockets, 'connect', fake_connect)
    attempts, success = await client.run_connect_attempts(max_attempts=4, cap_sleep=0.01)
    assert success is True
    assert attempts >= 3
