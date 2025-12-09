import asyncio
import os
import tempfile
import time

import pandas as pd
import pytest

from src.ingestion.providers.kraken_ws import KrakenWSClient


@pytest.mark.asyncio
async def test_duplicate_and_out_of_order_trades_are_ignored(tmp_path):
    out_root = str(tmp_path)
    client = KrakenWSClient(out_root=out_root)
    # run consumer only; avoid starting connect loop
    await client.start(start_connect_loop=False)

    pair = "XBTUSD"
    now = int(time.time())
    # first message: seq=1
    msg1 = {"type": "trade", "pair": pair, "timestamp": now, "price": 100.0, "size": 0.1, "seq": 1}
    await client.feed_message(msg1)
    # give consumer task time to process
    await asyncio.sleep(0.1)

    # buffer should contain one WAL entry for minute
    keys = list(client._wal_buffer.keys())
    assert keys, "wal buffer should have at least one key"
    key = keys[0]
    entries = client._wal_buffer.get(key, [])
    assert len(entries) == 1

    # duplicate message with same seq should be ignored
    dup = {"type": "trade", "pair": pair, "timestamp": now, "price": 100.0, "size": 0.1, "seq": 1}
    await client.feed_message(dup)
    await asyncio.sleep(0.1)
    entries2 = client._wal_buffer.get(key, [])
    assert len(entries2) == 1, "duplicate seq should not be appended"

    # out-of-order older timestamp should be ignored
    older = {"type": "trade", "pair": pair, "timestamp": now - 60, "price": 99.0, "size": 0.2, "seq": 0}
    await client.feed_message(older)
    await asyncio.sleep(0.1)
    entries3 = client._wal_buffer.get(key, [])
    assert len(entries3) == 1, "older timestamp trade should be ignored"

    # a new higher seq should be accepted
    msg2 = {"type": "trade", "pair": pair, "timestamp": now + 1, "price": 101.0, "size": 0.05, "seq": 2}
    await client.feed_message(msg2)
    await asyncio.sleep(0.1)

    # there may be a new minute key or same key depending on timestamps; ensure total entries >=2
    total = sum(len(v) for v in client._wal_buffer.values())
    assert total >= 2

    await client.stop()
