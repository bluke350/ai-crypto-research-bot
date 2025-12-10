import asyncio
import os
import pandas as pd
import pytest
import pyarrow.parquet as pq

from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_checkpoint_trades_handles_parquet_write_failure(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    pair = "XBTUSD"
    ts = int(pd.Timestamp.now(tz="UTC").timestamp())
    msg = {"type": "trade", "pair": pair, "timestamp": ts, "price": 123.45, "size": 0.5, "seq": 10}

    # monkeypatch pyarrow.parquet.write_table to raise to exercise the exception branch
    orig_write = pq.write_table

    def fake_write_table(*args, **kwargs):
        raise RuntimeError("fake write failure")

    monkeypatch.setattr(pq, 'write_table', fake_write_table)

    # run checkpoint (should not raise)
    asyncio.run(client._checkpoint_trades(msg))

    # wal buffer should still be updated (checkpoint path happened before write)
    minute_str = pd.to_datetime(ts, unit='s', utc=True).strftime("%Y%m%dT%H%M")
    key = (pair, minute_str)
    # since write_table failed, there may still be an in-memory entry
    assert key in client._wal_buffer

    # restore original
    monkeypatch.setattr(pq, 'write_table', orig_write)

