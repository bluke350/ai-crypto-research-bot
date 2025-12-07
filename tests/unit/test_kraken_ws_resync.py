import asyncio
import os
import pandas as pd
import pyarrow.parquet as pq
import pytest
from unittest.mock import patch

from src.ingestion.providers.kraken_ws import KrakenWSClient


@pytest.mark.asyncio
async def test_resync_writes_parquet(tmp_path):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    # create synthetic OHLC DataFrame that get_ohlc will return
    ts = pd.to_datetime(1609459200, unit='s', utc=True)
    df = pd.DataFrame([
        {"timestamp": ts + pd.Timedelta(minutes=0), "open": 29000.0, "high": 29100.0, "low": 28900.0, "close": 29050.0, "volume": 1.0, "count": 1},
        {"timestamp": ts + pd.Timedelta(minutes=1), "open": 29050.0, "high": 29150.0, "low": 29000.0, "close": 29100.0, "volume": 2.0, "count": 2},
    ])

    async def run_test():
        # avoid starting live websocket connect loop during unit tests
        await client.start(start_connect_loop=False)
        # simulate incoming message with seq=1
        await client.feed_message({"type": "trade", "pair": "XBT/USD", "timestamp": 1609459200, "price": 29000.0, "size": 0.001, "seq": 1})
        # now simulate a gap: next message has seq=3 (gap of 1)
        await client.feed_message({"type": "trade", "pair": "XBT/USD", "timestamp": 1609459260, "price": 29100.0, "size": 0.002, "seq": 3})
        # allow tasks to run
        await asyncio.sleep(0.5)
        await client.stop()

    # create synthetic trades DataFrame to return from get_trades
    trades = pd.DataFrame([
        {"timestamp": pd.to_datetime(1609459200, unit='s', utc=True), "price": 29000.0, "size": 0.001, "side": "b"},
        {"timestamp": pd.to_datetime(1609459260, unit='s', utc=True), "price": 29100.0, "size": 0.002, "side": "s"},
    ])
    # patch kraken_rest.get_trades so _resync_pair will use our trades
    with patch("src.ingestion.providers.kraken_ws.kraken_rest.get_trades", return_value=trades):
        await run_test()

    # check that resync parquet files exist for each minute
    day_dir = out / "XBT/USD" / pd.to_datetime(1609459200, unit='s').strftime("%Y%m%d")
    files = os.listdir(day_dir)
    assert any(f.endswith('.parquet') for f in files)
    # read back one parquet file and validate fields
    for f in files:
        if f.endswith('.parquet'):
            dfp = pq.read_table(os.path.join(day_dir, f)).to_pandas()
            assert 'vwap' in dfp.columns
            assert 'volume' in dfp.columns
            assert 'count' in dfp.columns
