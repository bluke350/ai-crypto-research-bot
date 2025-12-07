import asyncio
import os
import shutil
from datetime import datetime, timedelta
import time
import pytest
from src.ingestion.providers.kraken_ws import KrakenWSClient


@pytest.mark.asyncio
async def test_wal_compress_creates_tar_and_removes_dir(tmp_path):
    out = tmp_path / "data"
    os.environ["WS_WAL_COMPRESS_DAYS"] = "0"  # compress any day older than today
    os.environ["WS_WAL_COMPRESS_INTERVAL_HOURS"] = "0.001"  # run frequently

    client = KrakenWSClient(out_root=str(out))
    archive_dir = os.path.join(str(out), "_wal", "archive", "XBT/USD", (datetime.utcnow() - timedelta(days=2)).strftime("%Y%m%d"))
    os.makedirs(archive_dir, exist_ok=True)
    # add a dummy file
    with open(os.path.join(archive_dir, "dummy.parquet"), "w") as f:
        f.write("x")

    # call the synchronous helper directly for deterministic behavior in tests
    client.compress_archived_wal_once()

    # original dir should be removed and a .tar.gz should exist
    tar_path = archive_dir + ".tar.gz"
    assert not os.path.exists(archive_dir)
    assert os.path.exists(tar_path)

    os.environ.pop("WS_WAL_COMPRESS_DAYS", None)
    os.environ.pop("WS_WAL_COMPRESS_INTERVAL_HOURS", None)


@pytest.mark.asyncio
async def test_wal_compress_does_not_touch_today(tmp_path):
    out = tmp_path / "data"
    os.environ["WS_WAL_COMPRESS_DAYS"] = "1"  # compress days older than 1 day
    os.environ["WS_WAL_COMPRESS_INTERVAL_HOURS"] = "0.001"

    client = KrakenWSClient(out_root=str(out))
    # today's directory
    today_dir = os.path.join(str(out), "_wal", "archive", "XBT/USD", datetime.utcnow().strftime("%Y%m%d"))
    os.makedirs(today_dir, exist_ok=True)
    with open(os.path.join(today_dir, "live.parquet"), "w") as f:
        f.write("live")

    # first pass: should not compress today's dir
    client.compress_archived_wal_once()
    # simulate the client writing a new file into today's dir
    with open(os.path.join(today_dir, "more.parquet"), "w") as f:
        f.write("more")
    # second pass: still should not compress today's dir
    client.compress_archived_wal_once()

    # today's dir should still exist (not compressed)
    assert os.path.exists(today_dir)

    os.environ.pop("WS_WAL_COMPRESS_DAYS", None)
    os.environ.pop("WS_WAL_COMPRESS_INTERVAL_HOURS", None)
