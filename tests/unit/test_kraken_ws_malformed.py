import asyncio
import os
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.ingestion.providers.kraken_ws import KrakenWSClient


@pytest.mark.asyncio
async def test_checkpoint_trades_handles_non_numeric_fields(tmp_path):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))
    await client.start(start_connect_loop=False)
    # malformed price and size
    await client.feed_message({"type": "trade", "pair": "XBT/USD", "timestamp": 1609459200, "price": "not-a-number", "size": "not-a-number", "seq": 1})
    await asyncio.sleep(0.1)
    await client.stop()
    # WAL folder may exist, but there should be no parquet files created for this malformed message
    wal_dir = out / "_wal"
    if wal_dir.exists():
        found = False
        for root, dirs, files in os.walk(wal_dir):
            for f in files:
                if f.endswith('.parquet'):
                    found = True
                    break
            if found:
                break
        assert not found
