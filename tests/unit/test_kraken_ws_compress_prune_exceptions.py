import asyncio
import os
import shutil
from datetime import datetime, timedelta, timezone
import pytest
import pandas as pd
from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_compress_archived_wal_handles_make_archive_failure(tmp_path, monkeypatch):
    out = tmp_path / "data"
    os.environ["WS_WAL_COMPRESS_DAYS"] = "0"
    os.environ["WS_WAL_COMPRESS_INTERVAL_HOURS"] = "0.001"

    client = KrakenWSClient(out_root=str(out))
    archive_dir = os.path.join(str(out), "_wal", "archive", "XBT/USD", (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y%m%d"))
    os.makedirs(archive_dir, exist_ok=True)
    with open(os.path.join(archive_dir, "dummy.parquet"), "w") as f:
        f.write("x")

    # monkeypatch shutil.make_archive to raise
    orig_make = shutil.make_archive
    def fake_make_archive(*args, **kwargs):
        raise RuntimeError("fake make archive failure")

    monkeypatch.setattr(shutil, 'make_archive', fake_make_archive)

    # should not raise
    client.compress_archived_wal_once()

    # original dir should still exist due to failure
    assert os.path.exists(archive_dir)

    monkeypatch.setattr(shutil, 'make_archive', orig_make)
    os.environ.pop("WS_WAL_COMPRESS_DAYS", None)
    os.environ.pop("WS_WAL_COMPRESS_INTERVAL_HOURS", None)


def test_prune_archived_wal_handles_rmtree_failure(tmp_path, monkeypatch):
    out = tmp_path / "data"
    os.environ["WS_WAL_RETENTION_DAYS"] = "0"
    os.environ["WS_WAL_PRUNE_INTERVAL_HOURS"] = "0.001"

    client = KrakenWSClient(out_root=str(out))
    archive_root = os.path.join(str(out), "_wal", "archive", "XBT/USD")
    old_day = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y%m%d")
    old_dir = os.path.join(archive_root, old_day)
    os.makedirs(old_dir, exist_ok=True)
    with open(os.path.join(old_dir, "old.parquet"), "w") as f:
        f.write("old")

    # monkeypatch shutil.rmtree to raise when called so prune catches exception
    orig_rmtree = shutil.rmtree
    def fake_rmtree(path, *args, **kwargs):
        raise RuntimeError("fake rmtree failure")
    monkeypatch.setattr(shutil, 'rmtree', fake_rmtree)

    # should not raise
    client.prune_archived_wal_once()

    # old directory should still exist due to our fake rmtree failure
    assert os.path.exists(old_dir)

    monkeypatch.setattr(shutil, 'rmtree', orig_rmtree)
    os.environ.pop("WS_WAL_RETENTION_DAYS", None)
    os.environ.pop("WS_WAL_PRUNE_INTERVAL_HOURS", None)
