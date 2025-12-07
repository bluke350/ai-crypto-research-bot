import asyncio
import os
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_wal_flush_handles_parquet_write_error(monkeypatch, tmp_path):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))
    pair = "XBTUSD"
    ts = int(pd.Timestamp.utcnow().timestamp())
    minute_str = pd.to_datetime(ts, unit='s', utc=True).strftime("%Y%m%dT%H%M")
    key = (pair, minute_str)
    client._wal_buffer[key] = [{"timestamp": pd.to_datetime(pd.Timestamp.utcnow()), "price": 100.0, "size": 1.0}]

    # make pq.write_table raise on attempt to flush wal
    def fake_write_table(table, path):
        raise IOError("no space")

    monkeypatch.setattr(pq, 'write_table', fake_write_table)

    # call the synchronous flush helper - it should not raise
    asyncio.run(client._flush_wal_once())

    # if write failed, buffer for the key should be empty (it is popped before writing)
    # or was already popped; assert no unhandled exceptions and WAL dir may not exist
    wal_root = os.path.join(str(out), '_wal')
    # flush attempted, ensure archive not accidentally created
    assert os.path.exists(wal_root)
import asyncio
import os
import pandas as pd
import pytest
import pyarrow.parquet as pq

from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_wal_flush_handles_parquet_write_failure(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))
    client._wal_flush_interval = 0.05

    pair = "XBTUSD"
    ts = int(pd.Timestamp.utcnow().timestamp())
    minute_str = pd.to_datetime(ts, unit='s', utc=True).strftime("%Y%m%dT%H%M")
    key = (pair, minute_str)
    client._wal_buffer[key] = [{"timestamp": pd.to_datetime(ts, unit='s', utc=True), "price": 100.0, "size": 1.0}]

    # monkeypatch pq.write_table to raise
    orig_write = pq.write_table

    def fake_write_table(*args, **kwargs):
        raise RuntimeError("fake write failure")

    monkeypatch.setattr(pq, 'write_table', fake_write_table)

    async def runner():
        client._running = True
        task = asyncio.create_task(client._wal_flush_loop())
        await asyncio.sleep(0.2)
        client._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(runner())

    # because write failed, the WAL buffer has been popped (popped before write)
    # but no WAL files should have been created (write failed)
    assert key not in client._wal_buffer
    wal_root = os.path.join(str(out), '_wal')
    assert os.path.exists(wal_root)
    # ensure no parquet files present under the wal_root to indicate write did not succeed
    found = False
    for root, dirs, files in os.walk(wal_root):
        if any(f.endswith('.parquet') for f in files):
            found = True
            break
    assert not found

    monkeypatch.setattr(pq, 'write_table', orig_write)
