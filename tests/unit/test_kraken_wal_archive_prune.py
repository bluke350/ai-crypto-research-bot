import asyncio
import os
import tarfile
from datetime import datetime, timedelta
import pytest
from src.ingestion.providers.kraken_ws import KrakenWSClient


@pytest.mark.asyncio
async def test_wal_archive_prune_removes_old_tarballs(tmp_path):
    out = tmp_path / "data"
    os.environ["WS_WAL_ARCHIVE_COMPRESSION"] = "1"
    os.environ["WS_WAL_ARCHIVE_COMPRESS_AFTER_HOURS"] = "0"
    os.environ["WS_WAL_ARCHIVE_INTERVAL_HOURS"] = "0.001"
    # retention 0 days -> prune everything older than today
    os.environ["WS_WAL_RETENTION_DAYS"] = "0"
    os.environ["WS_WAL_PRUNE_INTERVAL_HOURS"] = "0.001"

    client = KrakenWSClient(out_root=str(out))
    # create a tarball with old mtime
    archive_root = os.path.join(str(out), "_wal", "archive", "XBT/USD")
    os.makedirs(archive_root, exist_ok=True)
    day = (datetime.utcnow() - timedelta(days=2)).strftime("%Y%m%d")
    tar_path = os.path.join(archive_root, day + ".tar.gz")
    with open(tar_path, "wb") as f:
        f.write(b"x")
    # set mtime to old
    old_ts = (datetime.utcnow() - timedelta(days=2)).timestamp()
    os.utime(tar_path, (old_ts, old_ts))

    await client.start()
    await asyncio.sleep(1.5)
    await client.stop()

    assert not os.path.exists(tar_path)

    # cleanup env
    for v in ["WS_WAL_ARCHIVE_COMPRESSION", "WS_WAL_ARCHIVE_COMPRESS_AFTER_HOURS", "WS_WAL_ARCHIVE_INTERVAL_HOURS", "WS_WAL_RETENTION_DAYS", "WS_WAL_PRUNE_INTERVAL_HOURS"]:
        os.environ.pop(v, None)
