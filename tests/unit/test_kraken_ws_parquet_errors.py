import asyncio
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import types
import pytest

from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_checkpoint_trades_handles_parquet_write_error(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    pair = "XBTUSD"
    ts = int(pd.Timestamp.now(tz="UTC").timestamp())
    msg = {"type": "trade", "pair": pair, "timestamp": ts, "price": 123.45, "size": 0.5, "seq": 10}

    # monkeypatch pq.write_table to raise an IOError
    orig_write = pq.write_table

    def fake_write(table, path, *args, **kwargs):
        raise IOError("disk full")

    monkeypatch.setattr(pq, 'write_table', fake_write)

    # should not raise; the function should handle the exception
    asyncio.run(client._checkpoint_trades(msg))

    # the temporary file should not exist
    minute = pd.to_datetime(ts, unit='s', utc=True).floor('min')
    minute_dt = pd.Timestamp(minute)
    out_dir = os.path.join(str(out), pair, minute_dt.strftime("%Y%m%d"))
    # either the dir does not exist, or it is empty
    if os.path.exists(out_dir):
        files = os.listdir(out_dir)
        assert not any(f.endswith('.parquet') for f in files)

    # restore original writer
    monkeypatch.setattr(pq, 'write_table', orig_write)
