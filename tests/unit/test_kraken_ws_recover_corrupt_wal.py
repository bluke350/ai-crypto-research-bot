import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone
import pytest

from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_recover_handles_corrupt_parquet(tmp_path, monkeypatch):
    out = tmp_path / 'data'
    client = KrakenWSClient(out_root=str(out))

    pair = 'XBTUSD'
    day = datetime.now(timezone.utc).strftime('%Y%m%d')
    wal_dir = os.path.join(client.wal_folder, pair, day)
    os.makedirs(wal_dir, exist_ok=True)

    # write a 'corrupt' file (non-parquet content with .parquet extension)
    path = os.path.join(wal_dir, 'corrupt.parquet')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('not a parquet file')

    # monkeypatch a no-op get_event_loop that uses call_soon_threadsafe inline
    class FakeLoop:
        def call_soon_threadsafe(self, cb):
            cb()

    import asyncio as _asyncio
    orig_get_loop = _asyncio.get_event_loop
    try:
        _asyncio.get_event_loop = lambda: FakeLoop()
        # recover should skip the corrupt file and continue
        client._recover_wal()
    finally:
        _asyncio.get_event_loop = orig_get_loop

    # after recover, corrupt file should remain (read failed) so no archive created
    archive_dir = os.path.join(client.wal_folder, 'archive', pair, day)
    assert not os.path.exists(archive_dir)
    # the corrupt file should still exist, and no messages enqueued
    assert os.path.exists(path)
    with pytest.raises(Exception):
        # no messages should have been enqueued
        client.msg_queue.get_nowait()
