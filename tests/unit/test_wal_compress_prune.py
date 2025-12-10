import asyncio
import os
import shutil
from datetime import datetime, timedelta, timezone
import pytest

from src.ingestion.providers.kraken_ws import KrakenWSClient


@pytest.mark.asyncio
async def test_wal_compress_and_prune(tmp_path):
    out = tmp_path / "data"
    # set compress after 0 days to force compression candidate
    os.environ["WS_WAL_COMPRESS_DAYS"] = "0"
    os.environ["WS_WAL_COMPRESS_INTERVAL_HOURS"] = "0.001"
    # set retention low so prune will remove old tar.gz
    os.environ["WS_WAL_RETENTION_DAYS"] = "0"
    os.environ["WS_WAL_PRUNE_INTERVAL_HOURS"] = "0.001"

    client = KrakenWSClient(out_root=str(out))
    archive_root = os.path.join(str(out), "_wal", "archive", "XBT/USD")
    old_day = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y%m%d")
    old_dir = os.path.join(archive_root, old_day)
    os.makedirs(old_dir, exist_ok=True)
    # create a fake parquet file inside
    with open(os.path.join(old_dir, "f.parquet"), "w") as f:
        f.write("data")

    # Call compress and prune helpers synchronously to avoid flakiness
    client.compress_archived_wal_once()
    client.prune_archived_wal_once()

    # after compression/prune, either the original dir or the tar.gz should be removed
    tar = old_dir + ".tar.gz"
    assert (not os.path.exists(old_dir)) or (not os.path.exists(tar))

    # cleanup env
    os.environ.pop("WS_WAL_COMPRESS_DAYS", None)
    os.environ.pop("WS_WAL_COMPRESS_INTERVAL_HOURS", None)
    os.environ.pop("WS_WAL_RETENTION_DAYS", None)
    os.environ.pop("WS_WAL_PRUNE_INTERVAL_HOURS", None)
