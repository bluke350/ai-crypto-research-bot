import asyncio
import os
import asyncio
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import pytest

from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_wal_flush_handles_write_error(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    pair = "XBTUSD"
    # prepare WAL buffer entry
    ts = int(pd.Timestamp.now(tz="UTC").timestamp())
    minute_str = pd.to_datetime(ts, unit='s', utc=True).strftime('%Y%m%dT%H%M')
    key = (pair, minute_str)
    client._wal_buffer[key] = [{"timestamp": pd.to_datetime(ts, unit='s', utc=True), "price": 100.0, "size": 1.0}]

    # monkeypatch pq.write_table to raise an IOError on write
    orig_write = pq.write_table

    def fake_write(table, path, *args, **kwargs):
        raise IOError("disk full")

    monkeypatch.setattr(pq, 'write_table', fake_write)

    async def runner():
        client._running = True
        # call flush loop body directly by invoking _flush_wal_once()
        await client._flush_wal_once()
        client._running = False

    # should not raise—error should be caught inside flush
    asyncio.run(runner())

    # buffer should have been emptied or kept depending on write behavior; ensure no crash and WAL dir not containing partial files
    wal_root = os.path.join(str(out), '_wal')
    if os.path.exists(wal_root):
        # check if any .tmp file exists -> none should exist
        for root, dirs, files in os.walk(wal_root):
            for f in files:
                assert not f.endswith('.tmp')

    # restore original writer
    monkeypatch.setattr(pq, 'write_table', orig_write)
