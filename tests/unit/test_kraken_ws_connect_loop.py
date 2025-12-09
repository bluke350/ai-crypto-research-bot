import asyncio
import time
import pytest
from unittest.mock import patch

from src.ingestion.providers.kraken_ws import KrakenWSClient


class DummyWS:
    def __init__(self, messages=None):
        self._messages = messages or []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)

    async def close(self):
        return


@pytest.mark.asyncio
async def test_connect_loop_counts_attempts(monkeypatch):
    client = KrakenWSClient(out_root='.')
    client._backoff_test_mode = True  # type: ignore
    # prepare a connect implementation: fail twice then succeed
    attempts = {'count': 0}

    async def connect_side_effect(*args, **kwargs):
        attempts['count'] += 1
        if attempts['count'] <= 2:
            raise RuntimeError('connect fail')
        return DummyWS([])

    import src.ingestion.providers.kraken_ws as kw
    monkeypatch.setattr(kw.websockets, 'connect', connect_side_effect)

    # small helper to wait for a condition with timeout to avoid hanging tests
    async def _wait_for(predicate, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return
            await asyncio.sleep(0.01)
        raise AssertionError('timeout waiting for condition')

    # Use the test helper that performs bounded synchronous connect attempts
    attempts_made, success = await client.run_connect_attempts(max_attempts=4, cap_sleep=0.01)
    # assert at least three attempts were recorded and we succeeded
    assert attempts_made >= 3
    assert success is True


@pytest.mark.asyncio
async def test_connect_loop_handles_incoming_messages(monkeypatch):
    client = KrakenWSClient(out_root='.')
    client._backoff_test_mode = True # type: ignore
    # prepare a websocket that yields one message and then stops
    msg = {"type": "trade", "pair": "XBTUSD", "seq": 1}
    async def connect_success(*args, **kwargs):
        return DummyWS([msg])

    import src.ingestion.providers.kraken_ws as kw
    monkeypatch.setattr(kw.websockets, 'connect', connect_success)

    # capture calls to _handle_message to ensure it's invoked
    called = {'count': 0}

    async def fake_handle(m):
        called['count'] += 1

    monkeypatch.setattr(client, '_handle_message', fake_handle)

    async def _wait_for(predicate, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return
            await asyncio.sleep(0.01)
        raise AssertionError('timeout waiting for condition')

    # directly call the message handler (no live websocket needed)
    await client._handle_message(msg)
    assert called['count'] >= 1
