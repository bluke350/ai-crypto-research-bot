import asyncio
import os
import shutil
import pytest
from src.ingestion.providers.kraken_ws import KrakenWSClient


@pytest.mark.asyncio
async def test_kraken_ws_checkpoint(tmp_path):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))
    await client.start()
    msg = {"type": "trade", "pair": "XBT/USD", "timestamp": 1609459200, "price": 29000.0, "size": 0.001}
    await client.feed_message(msg)
    # allow consumer to process
    await asyncio.sleep(0.2)
    await client.stop()
    # check that parquet file exists
    p = out / "XBT/USD" / "20210101" / "trades.parquet"
    assert p.exists()
