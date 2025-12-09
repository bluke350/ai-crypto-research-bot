import os
import pyarrow.parquet as pq
import pandas as pd
from datetime import datetime
import pytest

from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_recover_wal_handles_corrupt_parquet(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    pair = 'XBTUSD'
    day = datetime.utcnow().strftime('%Y%m%d')
    wal_dir = os.path.join(client.wal_folder, pair, day)
    os.makedirs(wal_dir, exist_ok=True)

    # create a corrupt parquet file (write invalid content)
    path = os.path.join(wal_dir, 'sample.parquet')
    with open(path, 'wb') as fh:
        fh.write(b"not a parquet")

    # monkeypatch pq.read_table to raise
    orig_read = pq.read_table

    def fake_read(p):
        raise IOError('corrupt parquet')

    monkeypatch.setattr(pq, 'read_table', fake_read)

    # monkeypatch loop to call enqueuer immediately
    class FakeLoop:
        def call_soon_threadsafe(self, cb):
            cb()

    import asyncio as _asyncio
    orig_get_loop = _asyncio.get_event_loop
    try:
        _asyncio.get_event_loop = lambda: FakeLoop()
        # should not raise
        client._recover_wal()
    finally:
        _asyncio.get_event_loop = orig_get_loop

    # ensure parquet file moved to archive (since _recover_wal moves files after reading)
    archive_dir = os.path.join(client.wal_folder, 'archive', pair, day)
    assert os.path.exists(archive_dir)

    monkeypatch.setattr(pq, 'read_table', orig_read)
import os

import pyarrow as pa
import pyarrow.parquet as pq

from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_recover_wal_handles_corrupt_parquet(tmp_path):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    pair = "XBTUSD"
    day = "20200101"
    wal_dir = os.path.join(client.wal_folder, pair, day)
    os.makedirs(wal_dir, exist_ok=True)

    # create a corrupt parquet file (not a valid parquet)
    p = os.path.join(wal_dir, "bad.parquet")
    with open(p, "w") as f:
        f.write("not a parquet")

    # monkeypatch asyncio.get_event_loop to a fake that calls callbacks synchronously
    class FakeLoop:
        def call_soon_threadsafe(self, cb):
            cb()

    import asyncio as _asyncio
    orig = _asyncio.get_event_loop
    try:
        _asyncio.get_event_loop = lambda: FakeLoop()
        # should not raise
        client._recover_wal()
    finally:
        _asyncio.get_event_loop = orig

    # corrupt file should still exist (recover attempted and logged the error)
    assert os.path.exists(p)
