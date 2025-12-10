import asyncio
import os
import pandas as pd
import pyarrow.parquet as pq

from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_checkpoint_trades_writes_minute_parquet_and_buffers_wal(tmp_path):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    pair = "XBTUSD"
    # create a timestamp in seconds
    ts = int(pd.Timestamp.now(tz="UTC").timestamp())
    msg = {"type": "trade", "pair": pair, "timestamp": ts, "price": 123.45, "size": 0.5, "seq": 10}

    # run checkpoint coroutine
    asyncio.run(client._checkpoint_trades(msg))

    # check last_ts updated
    assert client._last_ts.get(pair) == ts

    # check parquet minute file exists
    minute = pd.to_datetime(ts, unit='s', utc=True).floor('min')
    minute_dt = pd.Timestamp(minute)
    out_dir = os.path.join(str(out), pair, minute_dt.strftime("%Y%m%d"))
    files = os.listdir(out_dir)
    assert any(f.endswith('.parquet') for f in files)

    # check wal buffer has entry for the minute string
    minute_str = pd.to_datetime(ts, unit='s', utc=True).strftime("%Y%m%dT%H%M")
    key = (pair, minute_str)
    assert key in client._wal_buffer
    assert len(client._wal_buffer[key]) >= 1
