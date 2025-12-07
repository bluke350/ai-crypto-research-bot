import asyncio
import os
import shutil
from datetime import datetime, timedelta
import pytest
import pandas as pd
from src.ingestion.providers.kraken_ws import KrakenWSClient


@pytest.mark.asyncio
async def test_wal_prune_preserves_non_date_dirs(tmp_path):
    out = tmp_path / "data"
    os.environ["WS_WAL_RETENTION_DAYS"] = "0"  # aggressive prune
    os.environ["WS_WAL_PRUNE_INTERVAL_HOURS"] = "0.001"

    client = KrakenWSClient(out_root=str(out))
    archive_root = os.path.join(str(out), "_wal", "archive", "XBT/USD")
    # create a non-date directory inside archive
    misc_dir = os.path.join(archive_root, "misc_folder")
    os.makedirs(misc_dir, exist_ok=True)
    with open(os.path.join(misc_dir, "file.txt"), "w") as f:
        f.write("keep me")

    # call the synchronous prune helper directly for deterministic behavior in tests
    client.prune_archived_wal_once()

    # non-date dir should still exist (prune only targets YYYYMMDD dirs)
    assert os.path.exists(misc_dir)

    # cleanup env
    os.environ.pop("WS_WAL_RETENTION_DAYS", None)
    os.environ.pop("WS_WAL_PRUNE_INTERVAL_HOURS", None)


@pytest.mark.asyncio
async def test_wal_prune_respects_retention(tmp_path):
    out = tmp_path / "data"
    # keep only today (retention 1 day)
    os.environ["WS_WAL_RETENTION_DAYS"] = "1"
    os.environ["WS_WAL_PRUNE_INTERVAL_HOURS"] = "0.001"

    client = KrakenWSClient(out_root=str(out))
    archive_root = os.path.join(str(out), "_wal", "archive", "XBT/USD")
    # old day (2 days ago)
    old_day = (datetime.utcnow() - timedelta(days=2)).strftime("%Y%m%d")
    old_dir = os.path.join(archive_root, old_day)
    os.makedirs(old_dir, exist_ok=True)
    with open(os.path.join(old_dir, "old.parquet"), "w") as f:
        f.write("old")
    # recent day (today)
    recent_day = datetime.utcnow().strftime("%Y%m%d")
    recent_dir = os.path.join(archive_root, recent_day)
    os.makedirs(recent_dir, exist_ok=True)
    with open(os.path.join(recent_dir, "new.parquet"), "w") as f:
        f.write("new")

    # call the synchronous prune helper directly (deterministic)
    client.prune_archived_wal_once()

    # old_dir should be removed, recent_dir should remain
    assert not os.path.exists(old_dir)
    assert os.path.exists(recent_dir)

    # cleanup env
    os.environ.pop("WS_WAL_RETENTION_DAYS", None)
    os.environ.pop("WS_WAL_PRUNE_INTERVAL_HOURS", None)
