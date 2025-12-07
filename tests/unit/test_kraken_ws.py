import asyncio
import os
import shutil
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.ingestion.providers.kraken_ws import KrakenWSClient


@pytest.mark.asyncio
async def test_ws_buffering_writes_minute_file(tmp_path):
    out_root = str(tmp_path)
    client = KrakenWSClient(out_root=out_root)

    await client.start(start_connect_loop=False)
    # feed 3 trade messages in same minute for pair XBT/USD
    ts = 1609459200
    msgs = [
        {"type": "trade", "pair": "XBT/USD", "timestamp": ts, "price": 30000.0, "size": 0.1},
        {"type": "trade", "pair": "XBT/USD", "timestamp": ts + 10, "price": 30010.0, "size": 0.2},
        {"type": "trade", "pair": "XBT/USD", "timestamp": ts + 20, "price": 29990.0, "size": 0.1},
    ]
    for m in msgs:
        await client.feed_message(m)

    # allow consumer to process
    await asyncio.sleep(0.2)
    await client.stop()

    # check file exists
    day_dir = os.path.join(out_root, "XBT/USD", pd.to_datetime(1609459200, unit="s").strftime("%Y%m%d"))
    files = os.listdir(day_dir)
    assert any(f.endswith('.parquet') for f in files)
    # read parquet and validate vwap/volume/count
    for f in files:
        if f.endswith('.parquet'):
            df = pq.read_table(os.path.join(day_dir, f)).to_pandas()
            assert df['volume'].sum() == 0.4
            assert df['count'].sum() == 3


@pytest.mark.asyncio
async def test_kraken_ws_checkpoint(tmp_path):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))
    await client.start(start_connect_loop=False)
    msg = {"type": "trade", "pair": "XBT/USD", "timestamp": 1609459200, "price": 29000.0, "size": 0.001}
    await client.feed_message(msg)
    # allow consumer to process
    await asyncio.sleep(0.2)
    await client.stop()
    # check that at least one parquet file exists for the day (timestamped filename expected)
    day_dir = out / "XBT/USD" / "20210101"
    files = list(day_dir.glob("*.parquet")) if day_dir.exists() else []
    assert files, f"no parquet files found in {day_dir}"
