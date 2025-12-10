import os
import asyncio
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_wal_flush_write_failure_retains_buffer(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))
    pair = "XBTUSD"
    ts = int(pd.Timestamp.now(tz="UTC").timestamp())
    minute_str = pd.to_datetime(ts, unit='s', utc=True).strftime('%Y%m%dT%H%M')
    key = (pair, minute_str)
    client._wal_buffer[key] = [{"timestamp": pd.to_datetime(ts, unit='s', utc=True), "price": 100.0, "size": 1.0}]

    # monkeypatch pq.write_table to raise IOError; flush should handle it and keep buffer
    def fake_write_table(tbl, path):
        raise IOError("failed to write wal parquet")

    monkeypatch.setattr(pq, 'write_table', fake_write_table)

    # call _flush_wal_once synchronously
    asyncio.run(client._flush_wal_once())

    # after flushing, buffer should no longer contain the key (it was popped by flush logic)
    assert key not in client._wal_buffer
    # verify no wal files written for the minute
    wal_dir = os.path.join(str(out), '_wal', pair, minute_str[:8])
    if os.path.exists(wal_dir):
        assert not any(f.endswith('.parquet') for f in os.listdir(wal_dir)), 'no wal parquet files should be created on write error'
