import os
import shutil
from datetime import datetime, timedelta
import pytest

from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_compress_handles_shutil_errors(monkeypatch, tmp_path):
    out = tmp_path / "data"
    os.environ["WS_WAL_COMPRESS_DAYS"] = "0"
    client = KrakenWSClient(out_root=str(out))
    archive_dir = os.path.join(str(out), '_wal', 'archive', 'XBTUSD', (datetime.utcnow() - timedelta(days=2)).strftime('%Y%m%d'))
    os.makedirs(archive_dir, exist_ok=True)
    # add a dummy file
    with open(os.path.join(archive_dir, 'dummy.parquet'), 'w') as f:
        f.write('x')

    # make shutil.make_archive raise
    def fake_make_archive(base_name, format, root_dir=None):
        raise Exception('archive failed')

    monkeypatch.setattr(shutil, 'make_archive', fake_make_archive)

    # calling compress helper should not raise
    client.compress_archived_wal_once()

    os.environ.pop('WS_WAL_COMPRESS_DAYS', None)


def test_prune_handles_os_errors(monkeypatch, tmp_path):
    out = tmp_path / 'data'
    os.environ['WS_WAL_RETENTION_DAYS'] = '0'
    client = KrakenWSClient(out_root=str(out))
    archive_root = os.path.join(str(out), '_wal', 'archive')
    # create a day directory older than retention
    day = (datetime.utcnow() - timedelta(days=10)).strftime('%Y%m%d')
    day_dir = os.path.join(archive_root, 'XBTUSD', day)
    os.makedirs(day_dir, exist_ok=True)
    with open(os.path.join(day_dir, 'var.parquet'), 'w') as f:
        f.write('x')

    # simulate os.remove raising
    def fake_remove(p):
        raise PermissionError('read only')

    monkeypatch.setattr(os, 'remove', fake_remove)

    # prune should not raise
    client.prune_archived_wal_once()

    os.environ.pop('WS_WAL_RETENTION_DAYS', None)
