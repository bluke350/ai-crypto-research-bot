import asyncio

from src.ingestion.providers.kraken_ws import KrakenWSClient


class DummyWS:
    def __init__(self, messages):
        self._messages = messages

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)

    async def close(self):
        return


async def _run_connect_with_side_effect(client, side_effect):
    # patch websockets.connect in module under test
    import src.ingestion.providers.kraken_ws as kw

    kw.websockets.connect = side_effect
    return await client.run_connect_attempts(max_attempts=3, cap_sleep=0.01)


def test_run_connect_attempts_succeeds_after_failure(monkeypatch):
    out = '.'
    client = KrakenWSClient(out_root=out)

    calls = {'n': 0}

    async def side_effect(*args, **kwargs):
        calls['n'] += 1
        # fail first call, succeed on second
        if calls['n'] == 1:
            raise RuntimeError("connect failed")
        return DummyWS([])

    attempts, ok = asyncio.run(_run_connect_with_side_effect(client, side_effect))

    assert attempts >= 2
    assert ok is True
