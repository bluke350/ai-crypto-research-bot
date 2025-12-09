import asyncio
import os
import threading
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def test_wal_flush_loop_runs_one_iteration(tmp_path):
    from src.ingestion.providers.kraken_ws import KrakenWSClient

    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))
    pair = "XBTUSD"
    minute = pd.to_datetime(int(pd.Timestamp.utcnow().timestamp()), unit='s', utc=True).strftime("%Y%m%dT%H%M")
    key = (pair, minute)
    # populate wal buffer with one entry
    client._wal_buffer[key] = [{"timestamp": pd.to_datetime(pd.Timestamp.utcnow()), "price": 100.0, "size": 1.0}]

    async def runner():
        # run the loop as a background task, let it wake and flush once
        client._running = True
        task = asyncio.create_task(client._wal_flush_loop())
        # wait a short while for the loop to run (uses client's _wal_flush_interval)
        await asyncio.sleep(min(0.2, client._wal_flush_interval + 0.05))
        # stop and cancel
        client._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(runner())

    # check wal files created
    wal_root = os.path.join(str(out), '_wal')
    assert os.path.exists(wal_root)


def test_start_health_server_uses_httpserver(monkeypatch):
    from src.ingestion.providers.kraken_ws import KrakenWSClient

    created = {}

    class FakeHTTPServer:
        def __init__(self, addr, handler):
            created['addr'] = addr
            created['handler'] = handler

        def serve_forever(self):
            # don't block
            return

    # patch the HTTPServer used inside the kraken_ws module
    import src.ingestion.providers.kraken_ws as kw

    monkeypatch.setattr(kw, 'HTTPServer', FakeHTTPServer)

    client = KrakenWSClient(out_root='.')
    # start health server (will instantiate FakeHTTPServer instead of real one)
    client._start_health_server()

    # thread should be created
    assert client._health_thread is not None
    # our fake server should have been constructed with ('', port)
    assert 'addr' in created
    assert isinstance(created['addr'], tuple)
    assert created['addr'][1] == client._health_port
