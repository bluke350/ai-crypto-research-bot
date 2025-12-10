import os
import shutil
import tempfile
from datetime import timedelta, datetime, timezone
import pytest

from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_compress_handles_failures(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))
    # prepare an archive day older than compress threshold
        old_day = (datetime.now(timezone.utc) - timedelta(days=10)).strftime('%Y%m%d')
    archive_dir = os.path.join(str(out), '_wal', 'archive', 'XBT/USD', old_day)
    os.makedirs(archive_dir, exist_ok=True)
    with open(os.path.join(archive_dir, 'a.parquet'), 'w') as fh:
        fh.write('x')

    # monkeypatch shutil.make_archive to raise
    orig_make = shutil.make_archive

    def fake_make_archive(base, fmt, root_dir=None, **kwargs):
        raise IOError('permission denied')

    monkeypatch.setattr(shutil, 'make_archive', fake_make_archive)

    # should not raise
    client.compress_archived_wal_once()

    # ensure original dir still exists (no deletion without compression)
    assert os.path.exists(archive_dir)

    monkeypatch.setattr(shutil, 'make_archive', orig_make)


def test_prune_handles_failures(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))
    # create an old archive file
        old_day = (datetime.now(timezone.utc) - timedelta(days=10)).strftime('%Y%m%d')
    archive_dir = os.path.join(str(out), '_wal', 'archive', 'XBT/USD', old_day)
    os.makedirs(archive_dir, exist_ok=True)
    fname = os.path.join(archive_dir, 'a.parquet')
    with open(fname, 'w') as fh:
        fh.write('x')
    # monkeypatch os.remove to raise when attempting to remove a file
    import os as _os
    orig_remove = _os.remove

    def fake_remove(p):
        raise PermissionError('no permission')

    monkeypatch.setattr(_os, 'remove', fake_remove)

    # should not raise
    client.prune_archived_wal_once()

    # cleanup monkeypatch
    monkeypatch.setattr(_os, 'remove', orig_remove)
import os
import shutil
from datetime import datetime, timedelta
import pytest
from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_compress_archived_wal_handles_make_archive_failure(tmp_path, monkeypatch):
    out = tmp_path / "data"
    os.environ['WS_WAL_COMPRESS_DAYS'] = '0'
    client = KrakenWSClient(out_root=str(out))
    # create an old archived day dir
        old_day = (datetime.now(timezone.utc) - timedelta(days=3)).strftime('%Y%m%d')
    archive_dir = os.path.join(str(out), '_wal', 'archive', 'XBTUSD', old_day)
    os.makedirs(archive_dir, exist_ok=True)
    with open(os.path.join(archive_dir, 'dummy.parquet'), 'w') as f:
        f.write('x')

    # monkeypatch shutil.make_archive to raise
    def fake_make_archive(base_name, format, root_dir=None):
        raise IOError('failed to compress')

    monkeypatch.setattr(shutil, 'make_archive', fake_make_archive)

    # call helper; should not raise despite make_archive failure
    client.compress_archived_wal_once()

    # directory should still exist because compression failed
    assert os.path.exists(archive_dir)
    # cleanup env
    os.environ.pop('WS_WAL_COMPRESS_DAYS', None)


def test_prune_archived_wal_handles_deletion_failure(tmp_path, monkeypatch):
    out = tmp_path / 'data'
    client = KrakenWSClient(out_root=str(out))
    # create an old archived day dir and a tar.gz file
        old_day = (datetime.now(timezone.utc) - timedelta(days=3)).strftime('%Y%m%d')
    archive_root = os.path.join(str(out), '_wal', 'archive', 'XBTUSD')
    day_dir = os.path.join(archive_root, old_day)
    os.makedirs(day_dir, exist_ok=True)
    tar_path = day_dir + '.tar.gz'
    with open(tar_path, 'w') as f:
        f.write('t')

    # monkeypatch os.remove and shutil.rmtree to raise
    def fake_remove(p):
        raise PermissionError('cannot remove file')

    def fake_rmtree(p):
        raise PermissionError('cannot rmtree')

    monkeypatch.setattr(os, 'remove', fake_remove)
    monkeypatch.setattr(shutil, 'rmtree', fake_rmtree)

    # prune should not raise despite deletion errors
    client.prune_archived_wal_once()

    # files/folders should still exist
    assert os.path.exists(day_dir)
    assert os.path.exists(tar_path)
