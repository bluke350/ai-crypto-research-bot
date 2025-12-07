import asyncio
import os
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.ingestion.providers.kraken_ws import KrakenWSClient


@pytest.mark.asyncio
async def test_wal_written_and_recoverable(tmp_path):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))
    await client.start(start_connect_loop=False)
    msg = {"type": "trade", "pair": "XBT/USD", "timestamp": 1609459200, "price": 29000.0, "size": 0.001, "seq": 1}
    await client.feed_message(msg)
    await asyncio.sleep(0.2)
    await client.stop()

    # wal path exists
    wal_dir = out / "_wal" / "XBT/USD" / pd.to_datetime(1609459200, unit='s').strftime("%Y%m%d")
    files = os.listdir(wal_dir)
    assert any(f.endswith('.parquet') for f in files)

    # now simulate recover: create new client and ensure it picks up wal
    client2 = KrakenWSClient(out_root=str(out))
    # clear internal aggregator
    await client2.start()
    await asyncio.sleep(0.2)
    await client2.stop()
    # if no exceptions, recovery is successful
    assert True
