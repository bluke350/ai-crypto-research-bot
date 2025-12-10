import asyncio
import os
import shutil
import types
from datetime import datetime, timedelta, timezone

import pandas as pd
"""Tests for WAL robustness and archive handling in KrakenWSClient.

These were normalized to use timezone-aware timestamps via the test helper.
"""

import asyncio
import os
import shutil
from datetime import timedelta

import pandas as pd
import pyarrow.parquet as pq

from src.ingestion.providers.kraken_ws import KrakenWSClient
from src.utils.time import now_utc


def test_checkpoint_trades_handles_parquet_write_failure(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    pair = "XBTUSD"
    ts = int(pd.Timestamp.now(tz="UTC").timestamp())
    msg = {"type": "trade", "pair": pair, "timestamp": ts, "price": 123.45, "size": 0.5, "seq": 10}

    # monkeypatch parquet write to raise
    import src.ingestion.providers.kraken_ws as kw

    def bad_write(table, path):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(kw.pq, "write_table", bad_write)

    # ensure no exception is raised and WAL buffer still records the trade
    asyncio.run(client._checkpoint_trades(msg))

    # wal buffer should have been populated despite parquet write failure
    minute_str = pd.to_datetime(ts, unit='s', utc=True).strftime("%Y%m%dT%H%M")
    key = (pair, minute_str)
    assert key in client._wal_buffer
    assert len(client._wal_buffer[key]) >= 1


def test_wal_flush_handles_parquet_write_failure(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    pair = "XBTUSD"
    minute_str = pd.to_datetime(pd.Timestamp.now(tz="UTC"), utc=True).strftime("%Y%m%dT%H%M")
    key = (pair, minute_str)
    client._wal_buffer[key] = [{"timestamp": pd.to_datetime(pd.Timestamp.now(tz="UTC")), "price": 100.0, "size": 1.0}]

    # monkeypatch pq.write_table to fail
    import src.ingestion.providers.kraken_ws as kw

    def bad_write(table, path):
        raise RuntimeError("flush failure")

    monkeypatch.setattr(kw.pq, "write_table", bad_write)

    # call flush once; should not raise
    asyncio.run(client._flush_wal_once())

    # after flush attempt, wal buffer may be empty (pop on process), but no parquet file should exist
    wal_dir = os.path.join(str(out), '_wal', pair)
    if os.path.exists(wal_dir):
        files = [f for f in os.listdir(wal_dir) if f.endswith('.parquet')]
        assert len(files) == 0


def test_compress_archived_wal_handles_make_archive_failure(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    # create an archive directory older than threshold
    past_day = (now_utc() - timedelta(days=5)).strftime('%Y%m%d')
    archive_dir = os.path.join(str(out), '_wal', 'archive', 'XBTUSD', past_day)
    os.makedirs(archive_dir, exist_ok=True)
    with open(os.path.join(archive_dir, 'dummy.parquet'), 'w') as fh:
        fh.write('x')

    import src.ingestion.providers.kraken_ws as kw

    def bad_make_archive(base_name, format, root_dir=None, base_dir=None):
        raise RuntimeError("compress failure")

    monkeypatch.setattr(kw.shutil, 'make_archive', bad_make_archive)

    # should not raise; the dir should still exist after failure
    client.compress_archived_wal_once()
    assert os.path.exists(archive_dir)


def test_prune_archived_wal_handles_rmtree_failure(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    # create a day directory older than retention
    old_day = (now_utc() - timedelta(days=10)).strftime('%Y%m%d')
    archive_dir = os.path.join(str(out), '_wal', 'archive', 'XBTUSD', old_day)
    os.makedirs(archive_dir, exist_ok=True)

    import src.ingestion.providers.kraken_ws as kw

    def bad_rmtree(path):
        raise RuntimeError("rmtree failure")

    monkeypatch.setattr(kw.shutil, 'rmtree', bad_rmtree)

    # should not raise; dir should still exist
    client.prune_archived_wal_once()
    assert os.path.exists(archive_dir)


def test_recover_wal_handles_corrupt_parquet(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))
    pair = 'XBTUSD'
    day = now_utc().strftime('%Y%m%d')
    wal_dir = os.path.join(client.wal_folder, pair, day)
    os.makedirs(wal_dir, exist_ok=True)

    # write a dummy (non-parquet) file but with .parquet extension to simulate corruption
    path = os.path.join(wal_dir, 'bad.parquet')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('not a parquet')

    # monkeypatch pq.read_table to raise when reading this file
    import src.ingestion.providers.kraken_ws as kw

    def bad_read_table(p):
        raise RuntimeError('corrupt parquet')

    monkeypatch.setattr(kw.pq, 'read_table', lambda p: bad_read_table(p))

    # run recovery; should not raise and original file should remain
    client._recover_wal()
    assert os.path.exists(path)
