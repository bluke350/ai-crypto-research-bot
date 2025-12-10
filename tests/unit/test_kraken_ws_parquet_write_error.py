import asyncio
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from unittest import mock

from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_checkpoint_trades_parquet_write_failure(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))
    pair = "XBTUSD"
    ts = int(pd.Timestamp.now(tz="UTC").timestamp())
    msg = {"type": "trade", "pair": pair, "timestamp": ts, "price": 123.45, "size": 0.5, "seq": 10}

    # monkeypatch pq.write_table to raise an IOError to simulate disk/write failure
    def fake_write_table(tbl, path):
        raise IOError("disk write error")

    monkeypatch.setattr(pq, "write_table", fake_write_table)

    # run checkpoint; it should handle the exception and not crash
    asyncio.run(client._checkpoint_trades(msg))

    # ensure no tmp or partial file exists under the minute folder
    minute = pd.to_datetime(ts, unit='s', utc=True).floor('min')
    out_dir = os.path.join(str(out), pair, minute.strftime('%Y%m%d'))
    # directory might or might not exist depending if exception happened prior to os.makedirs
    if os.path.exists(out_dir):
        files = os.listdir(out_dir)
        assert not any(f.endswith('.tmp') for f in files), 'tmp files should not be left behind on write failure'

    # WAL buffer should still contain the entry since write failed
    minute_str = pd.to_datetime(ts, unit='s', utc=True).strftime('%Y%m%dT%H%M')
    key = (pair, minute_str)
    assert key in client._wal_buffer
