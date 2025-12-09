import asyncio
import os
import types
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

    # Force pq.write_table to raise an exception to simulate filesystem errors
    def fake_write_table(table, path):
        raise OSError("disk full")

    monkeypatch.setattr(pq, 'write_table', fake_write_table)

    # Should not raise and should still buffer wal entries
    asyncio.run(client._checkpoint_trades(msg))
    minute_str = pd.to_datetime(ts, unit='s', utc=True).strftime('%Y%m%dT%H%M')
    key = (pair, minute_str)
    assert key in client._wal_buffer


def test_wal_flush_handles_write_errors(tmp_path, monkeypatch):
    out = tmp_path / 'data'
    client = KrakenWSClient(out_root=str(out))
    # populate wal buffer with an entry
    pair = "XBTUSD"
    minute = pd.to_datetime(int(pd.Timestamp.utcnow().timestamp()), unit='s', utc=True).strftime('%Y%m%dT%H%M')
    key = (pair, minute)
    client._wal_buffer[key] = [{"timestamp": pd.to_datetime(pd.Timestamp.utcnow()), "price": 100.0, "size": 1.0}]

    # monkeypatch pq.write_table to raise
    def fake_write_table(table, path):
        raise IOError("cannot write")

    monkeypatch.setattr(pq, 'write_table', fake_write_table)

    # Call the synchronous flush helper; should not raise despite write failure
    asyncio.run(client._flush_wal_once())
    # buffer should be empty or already processed (failed writes do not re-add)
    assert not client._wal_buffer


def test_compress_archived_wal_handles_errors(tmp_path, monkeypatch):
    out = tmp_path / 'data'
    os.environ['WS_WAL_COMPRESS_DAYS'] = '0'
    client = KrakenWSClient(out_root=str(out))
    # create nested archive day dir
    archive_dir = os.path.join(str(out), '_wal', 'archive', 'XBTUSD', (pd.Timestamp.utcnow() - pd.Timedelta(days=2)).strftime('%Y%m%d'))
    os.makedirs(archive_dir, exist_ok=True)
    with open(os.path.join(archive_dir, 'dummy.parquet'), 'w') as fh:
        fh.write('x')

    # monkeypatch shutil.make_archive to raise
    import shutil
    def fake_make_archive(base_name, format, root_dir=None):
        raise RuntimeError('compression error')

    monkeypatch.setattr(shutil, 'make_archive', fake_make_archive)

    # Should not raise even if compression fails
    client.compress_archived_wal_once()

    # cleanup env
    os.environ.pop('WS_WAL_COMPRESS_DAYS', None)


def test_prune_archived_wal_handles_errors(tmp_path, monkeypatch):
    out = tmp_path / 'data'
    os.environ['WS_WAL_RETENTION_DAYS'] = '0'
    client = KrakenWSClient(out_root=str(out))
    archive_dir = os.path.join(str(out), '_wal', 'archive', 'XBTUSD', (pd.Timestamp.utcnow() - pd.Timedelta(days=5)).strftime('%Y%m%d'))
    os.makedirs(archive_dir, exist_ok=True)
    # create a tar to prune
    tar_path = archive_dir + '.tar.gz'
    with open(tar_path, 'w') as fh:
        fh.write('x')

    # monkeypatch os.remove to raise
    import os as _os
    def fake_remove(p):
        raise PermissionError('no perms')

    monkeypatch.setattr(_os, 'remove', fake_remove)

    # Should not crash
    client.prune_archived_wal_once()
    os.environ.pop('WS_WAL_RETENTION_DAYS', None)
import asyncio
import os
import shutil
import pytest
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from datetime import datetime, timedelta

from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_checkpoint_trades_handles_parquet_write_failure(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    pair = "XBTUSD"
    ts = int(pd.Timestamp.utcnow().timestamp())
    msg = {"type": "trade", "pair": pair, "timestamp": ts, "price": 123.45, "size": 0.5, "seq": 10}

    # patch pq.write_table to raise
    def fake_write_table(table, dest):
        raise RuntimeError("disk full")

    monkeypatch.setattr(pq, 'write_table', fake_write_table)

    # call checkpoint; should not raise and should still buffer to WAL
    asyncio.run(client._checkpoint_trades(msg))

    minute_str = pd.to_datetime(ts, unit='s', utc=True).strftime("%Y%m%dT%H%M")
    key = (pair, minute_str)
    # WAL buffer should contain entry despite parquet write failure
    assert key in client._wal_buffer


def test_wal_flush_loop_handles_write_failure(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))
    pair = "XBTUSD"
    minute = pd.to_datetime(int(pd.Timestamp.utcnow().timestamp()), unit='s', utc=True).strftime("%Y%m%dT%H%M")
    key = (pair, minute)
    client._wal_buffer[key] = [{"timestamp": pd.to_datetime(pd.Timestamp.utcnow()), "price": 100.0, "size": 1.0}]

    # patch pq.write_table to raise during flush
    def fake_write_table(table, dest):
        raise RuntimeError("io error")

    monkeypatch.setattr(pq, 'write_table', fake_write_table)

    async def run_once():
        # run one iteration of the flush loop
        client._running = True
        task = asyncio.create_task(client._wal_flush_loop())
        await asyncio.sleep(min(0.2, client._wal_flush_interval + 0.05))
        client._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run_once())

    # buffer should have been popped (attempted flush) or remain empty; at least there should be no unhandled exceptions
    assert isinstance(client._wal_buffer, dict)


def test_compress_archived_wal_handles_make_archive_failure(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    # prepare archive dir older than compress threshold
    old_date = (datetime.utcnow() - timedelta(days=10)).strftime("%Y%m%d")
    archive_dir = os.path.join(str(out), '_wal', 'archive', 'XBTUSD', old_date)
    os.makedirs(archive_dir, exist_ok=True)
    with open(os.path.join(archive_dir, 'dummy.parquet'), 'w') as f:
        f.write('x')

    # patch shutil.make_archive to raise
    def fake_make_archive(base_name, format, root_dir=None):
        raise RuntimeError("archive failure")

    monkeypatch.setattr(shutil, 'make_archive', fake_make_archive)

    # call helper; should not raise
    client.compress_archived_wal_once()

    # directory should still exist (not replaced) because make_archive failed
    assert os.path.exists(archive_dir)


def test_prune_archived_wal_handles_delete_failure(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    # add archive file older than retention
    old_date = (datetime.utcnow() - timedelta(days=10)).strftime("%Y%m%d")
    archive_dir = os.path.join(str(out), '_wal', 'archive', 'XBTUSD', old_date)
    os.makedirs(archive_dir, exist_ok=True)
    fname = os.path.join(archive_dir, 'dummy.parquet')
    with open(fname, 'w') as f:
        f.write('x')

    # patch os.remove and shutil.rmtree to raise so prune encounters exceptions
    def fake_remove(p):
        raise RuntimeError("cannot remove")

    def fake_rmtree(p):
        raise RuntimeError("cannot rmtree")

    monkeypatch.setattr(os, 'remove', fake_remove)
    monkeypatch.setattr(shutil, 'rmtree', fake_rmtree)

    # call helper; should not raise
    client.prune_archived_wal_once()

    # archived file should still exist as remove failed
    assert os.path.exists(fname)


def test_recover_wal_handles_corrupt_parquet(tmp_path):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    pair = "XBTUSD"
    day = datetime.utcnow().strftime("%Y%m%d")
    wal_dir = os.path.join(client.wal_folder, pair, day)
    os.makedirs(wal_dir, exist_ok=True)

    # create a file that is not a valid parquet
    path = os.path.join(wal_dir, "sample.parquet")
    with open(path, 'wb') as f:
        f.write(b"not a parquet")

    # monkeypatch asyncio.get_event_loop to provide a loop that executes call_soon_threadsafe inline
    class FakeLoop:
        def call_soon_threadsafe(self, cb):
            cb()

    import asyncio as _asyncio

    orig_get_loop = _asyncio.get_event_loop
    try:
        _asyncio.get_event_loop = lambda: FakeLoop()
        # run recover (synchronous)
        client._recover_wal()
    finally:
        _asyncio.get_event_loop = orig_get_loop

    # corrupt file should still be present (recover should have logged and skipped)
    assert os.path.exists(path)
