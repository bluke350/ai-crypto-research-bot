import asyncio
import os
import json
import types
from datetime import datetime
import pandas as pd
import pyarrow.parquet as pq

import pytest

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


@pytest.mark.asyncio
async def test_start_connect_and_consume_writes_parquet_and_wal(tmp_path, monkeypatch):
    out = tmp_path / "data"
    # ensure health server does not start a real HTTPServer
    class FakeHTTPServer:
        def __init__(self, addr, handler):
            self.addr = addr
            self.handler = handler

        def serve_forever(self):
            return

    import src.ingestion.providers.kraken_ws as kw
    monkeypatch.setattr(kw, 'HTTPServer', FakeHTTPServer)

    client = KrakenWSClient(out_root=str(out))
    # disable long sleeps
    client._backoff_test_mode = True
    # set a short WAL flush interval so we see files quickly
    client._wal_flush_interval = 0.05

    now = int(pd.Timestamp.utcnow().timestamp())
    # craft raw JSON messages as the WS yields
    msg1 = json.dumps({"type": "trade", "pair": "XBT/USD", "timestamp": now, "price": "40000", "size": "0.001", "seq": 1})
    msg2 = json.dumps({"type": "trade", "pair": "XBT/USD", "timestamp": now + 60, "price": "40010", "size": "0.002", "seq": 2})
    # include a gap to trigger resync path
    msg3 = json.dumps({"type": "trade", "pair": "XBT/USD", "timestamp": now + 120, "price": "40020", "size": "0.001", "seq": 5})

    async def fake_connect(url, *args, **kwargs):
        return DummyWS([msg1, msg2, msg3])

    monkeypatch.setattr('src.ingestion.providers.kraken_ws.websockets.connect', fake_connect)

    # patch kraken_rest.get_trades to return an empty DataFrame for resync (exercise branch)
    monkeypatch.setattr('src.ingestion.providers.kraken_ws.kraken_rest', types.SimpleNamespace(get_trades=lambda pair, s, e: pd.DataFrame()))

    # monkeypatch Event Loop create_task to capture resync tasks if needed
    created = []
    orig_create_task = asyncio.create_task

    def create_task(coro):
        t = orig_create_task(coro)
        created.append(t)
        return t

    monkeypatch.setattr(asyncio, 'create_task', create_task)

    # start client with connect loop and let it run briefly
    await client.start(start_connect_loop=True)
    await asyncio.sleep(0.3)
    # stop and flush
    await client.stop()

    # verify parquet files created for per-minute aggregation
    # find minute dir for the first message
    minute = pd.to_datetime(now, unit='s', utc=True).floor('min')
    day_dir = os.path.join(str(out), 'XBT/USD', minute.strftime('%Y%m%d'))
    assert os.path.exists(day_dir)
    files = [f for f in os.listdir(day_dir) if f.endswith('.parquet')]
    assert len(files) >= 1

    # verify WAL archive or raw files exist in _wal
    wal_root = os.path.join(str(out), '_wal')
    assert os.path.exists(wal_root)

    # ensure resync task scheduled (because of sequence gap)
    assert any(t for t in created if not t.cancelled())

