import asyncio
import os
import shutil
import types
from datetime import datetime, timedelta

import pandas as pd
import pyarrow.parquet as pq

from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_checkpoint_trades_handles_parquet_write_error(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    pair = "XBTUSD"
    ts = int(pd.Timestamp.utcnow().timestamp())
    msg = {"type": "trade", "pair": pair, "timestamp": ts, "price": 123.45, "size": 0.5, "seq": 10}

    # make parquet write raise
    orig_write = pq.write_table

    def fake_write(table, path):
        raise RuntimeError("disk error")

    monkeypatch.setattr(pq, 'write_table', fake_write)

    try:
        asyncio.run(client._checkpoint_trades(msg))
    finally:
        monkeypatch.setattr(pq, 'write_table', orig_write)

    # parquet minute file not created
    minute = pd.to_datetime(ts, unit='s', utc=True).floor('min')
    minute_dt = pd.Timestamp(minute)
    out_dir = os.path.join(str(out), pair, minute_dt.strftime('%Y%m%d'))
    assert not os.path.exists(out_dir) or not any(f.endswith('.parquet') for f in os.listdir(out_dir))

    # WAL buffer should still have entries appended despite parquet write failure
    minute_str = pd.to_datetime(ts, unit='s', utc=True).strftime('%Y%m%dT%H%M')
    key = (pair, minute_str)
    assert key in client._wal_buffer


def test_wal_flush_handles_parquet_write_error(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    pair = "XBTUSD"
    minute = pd.to_datetime(int(pd.Timestamp.utcnow().timestamp()), unit='s', utc=True).strftime('%Y%m%dT%H%M')
    key = (pair, minute)
    # populate wal buffer with one entry
    client._wal_buffer[key] = [{"timestamp": pd.to_datetime(pd.Timestamp.utcnow()), "price": 100.0, "size": 1.0}]

    # monkeypatch write_table to raise during WAL flush
    orig_write = pq.write_table

    def fake_write(table, path):
        raise RuntimeError("disk full")

    monkeypatch.setattr(pq, 'write_table', fake_write)

    try:
        # run flush loop once
        asyncio.run(client._flush_wal_once())
    finally:
        monkeypatch.setattr(pq, 'write_table', orig_write)

    # after flush, the buffer key should have been popped (the implementation pops before writing)
    assert key not in client._wal_buffer
    # wal folder should not contain the file (write failed)
    wal_root = os.path.join(str(out), '_wal')
    if os.path.exists(wal_root):
        # there should be no files under pair/day
        exists = False
        for root, dirs, files in os.walk(wal_root):
            for f in files:
                if f.endswith('.parquet'):
                    exists = True
        assert not exists


def test_compress_archived_wal_handles_exception(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    # create an archive directory older than compress threshold
    old_day = (datetime.utcnow() - timedelta(days=3)).strftime('%Y%m%d')
    archive_dir = os.path.join(str(out), '_wal', 'archive', 'XBT/USD', old_day)
    os.makedirs(archive_dir, exist_ok=True)
    with open(os.path.join(archive_dir, 'foo.parquet'), 'w') as fh:
        fh.write('x')

    # make shutil.make_archive raise
    orig_make = shutil.make_archive

    def fake_make_archive(base_name, fmt, root_dir=None):
        raise RuntimeError('compress failed')

    monkeypatch.setattr(shutil, 'make_archive', fake_make_archive)
    try:
        client.compress_archived_wal_once()
    finally:
        monkeypatch.setattr(shutil, 'make_archive', orig_make)

    # original directory should still exist (not removed because compress failed)
    assert os.path.exists(archive_dir)


def test_prune_archived_wal_handles_remove_exception(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    # create archive file older than retention
    old_day = (datetime.utcnow() - timedelta(days=10)).strftime('%Y%m%d')
    archive_dir = os.path.join(str(out), '_wal', 'archive', 'XBT/USD', old_day)
    os.makedirs(archive_dir, exist_ok=True)
    tar_path = os.path.join(archive_dir, 'old.tar.gz')
    with open(tar_path, 'w') as fh:
        fh.write('tar')
    # alter mtime to be old
    os.utime(tar_path, (0, 0))

    # monkeypatch os.remove and shutil.rmtree to raise when called to simulate failures
    orig_remove = os.remove
    orig_rmtree = shutil.rmtree

    def fake_remove(path):
        raise RuntimeError('delete failed')

    def fake_rmtree(path):
        raise RuntimeError('rmtree failed')

    monkeypatch.setattr(os, 'remove', fake_remove)
    monkeypatch.setattr(shutil, 'rmtree', fake_rmtree)
    try:
        # run prune
        client.prune_archived_wal_once()
    finally:
        monkeypatch.setattr(os, 'remove', orig_remove)
        monkeypatch.setattr(shutil, 'rmtree', orig_rmtree)

    # file should still exist because removal failed
    assert os.path.exists(tar_path)


def test_recover_wal_handles_corrupt_parquet(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    pair = 'XBTUSD'
    day = datetime.utcnow().strftime('%Y%m%d')
    wal_dir = os.path.join(client.wal_folder, pair, day)
    os.makedirs(wal_dir, exist_ok=True)
    path = os.path.join(wal_dir, 'bad.parquet')
    with open(path, 'wb') as f:
        f.write(b'not a parquet file')

    # monkeypatch pq.read_table to raise when called for this path
    orig_read = pq.read_table

    def fake_read_table(p):
        raise RuntimeError('corrupt parquet')

    monkeypatch.setattr(pq, 'read_table', fake_read_table)
    try:
        client._recover_wal()
    finally:
        monkeypatch.setattr(pq, 'read_table', orig_read)

    # file should still exist (not moved to archive)
    assert os.path.exists(path)
import os
import shutil
from datetime import datetime, timedelta
import pytest
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_checkpoint_trades_handles_parquet_write_failure(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    pair = "XBTUSD"
    ts = int(pd.Timestamp.utcnow().timestamp())
    msg = {"type": "trade", "pair": pair, "timestamp": ts, "price": 123.45, "size": 0.5, "seq": 10}

    # monkeypatch parquet write to raise
    import src.ingestion.providers.kraken_ws as kw

    def bad_write(table, path):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(kw.pq, "write_table", bad_write)

    # ensure no exception is raised and WAL buffer still records the trade
    import asyncio
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
    minute_str = pd.to_datetime(pd.Timestamp.utcnow(), utc=True).strftime("%Y%m%dT%H%M")
    key = (pair, minute_str)
    client._wal_buffer[key] = [{"timestamp": pd.to_datetime(pd.Timestamp.utcnow()), "price": 100.0, "size": 1.0}]

    # monkeypatch pq.write_table to fail
    import src.ingestion.providers.kraken_ws as kw

    def bad_write(table, path):
        raise RuntimeError("flush failure")

    monkeypatch.setattr(kw.pq, "write_table", bad_write)

    # call flush once; should not raise
    import asyncio
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
    past_day = (datetime.utcnow() - timedelta(days=5)).strftime("%Y%m%d")
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
    old_day = (datetime.utcnow() - timedelta(days=10)).strftime("%Y%m%d")
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
    day = datetime.utcnow().strftime('%Y%m%d')
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
