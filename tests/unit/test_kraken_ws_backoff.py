import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from src.ingestion.providers.kraken_ws import KrakenWSClient


class DummyWS:
    def __init__(self, messages):
        self._messages = messages
        self._closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)

    async def close(self):
        self._closed = True


@pytest.mark.asyncio
async def test_backoff_reconnect(tmp_path):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    # simulate websockets.connect failing twice, then succeeding with a DummyWS that yields no messages
    async def failing_connect(*args, **kwargs):
        raise RuntimeError("connect failed")

    async def succeeding_connect(*args, **kwargs):
        return DummyWS([])

    connect_calls = []

    async def connect_side_effect(*args, **kwargs):
        # first two calls fail, third succeeds
        connect_calls.append(1)
        if len(connect_calls) <= 2:
            raise RuntimeError("connect failed")
        return DummyWS([])

    with patch("src.ingestion.providers.kraken_ws.websockets.connect", new=connect_side_effect):
        # use the test helper to run bounded connect attempts deterministically
        attempts, ok = await client.run_connect_attempts(max_attempts=5, cap_sleep=0.05)

    # ensure at least 3 connect attempts were made (connect_side_effect increments connect_calls)
    assert len(connect_calls) >= 3
