import asyncio
import os
import pandas as pd
import pyarrow.parquet as pq
import pytest
import types

from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_checkpoint_trades_parquet_write_raises(monkeypatch, tmp_path):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    pair = "XBTUSD"
    ts = int(pd.Timestamp.utcnow().timestamp())
    msg = {"type": "trade", "pair": pair, "timestamp": ts, "price": 123.45, "size": 0.5, "seq": 10}

    # monkeypatch pq.write_table to raise an exception to simulate an I/O error
    def fake_write_table(table, path):
        raise IOError("disk full")

    monkeypatch.setattr(pq, 'write_table', fake_write_table)

    # run checkpoint coroutine and ensure it does not raise despite write failure
    asyncio.run(client._checkpoint_trades(msg))

    # even though parquet write failed, WAL buffer should still have an entry
    minute_str = pd.to_datetime(ts, unit='s', utc=True).strftime("%Y%m%dT%H%M")
    key = (pair, minute_str)
    assert key in client._wal_buffer and len(client._wal_buffer[key]) >= 1
