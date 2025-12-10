import asyncio
import os
import shutil
from datetime import datetime, timedelta, timezone
import time
import pytest
import pandas as pd
from src.ingestion.providers.kraken_ws import KrakenWSClient


@pytest.mark.asyncio
async def test_wal_prune_removes_old_archives(tmp_path):
    out = tmp_path / "data"
    os.environ["WS_WAL_RETENTION_DAYS"] = "0"  # prune any day older than today
    os.environ["WS_WAL_PRUNE_INTERVAL_HOURS"] = "0.001"  # run frequently (~3.6s)

    client = KrakenWSClient(out_root=str(out))
    # create fake archive folder with an old day
    archive_dir = os.path.join(str(out), "_wal", "archive", "XBT/USD", (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y%m%d"))
    os.makedirs(archive_dir, exist_ok=True)
    # create a dummy file in the archived day
    with open(os.path.join(archive_dir, "dummy.parquet"), "w") as f:
        f.write("x")

    await client.start(start_connect_loop=False)
    # wait enough for prune loop to run once
    await asyncio.sleep(1.5)
    # stop client and ensure prune task is cancelled
    await client.stop()

    # archived day should be removed
    assert not os.path.exists(archive_dir)

    # cleanup env
    os.environ.pop("WS_WAL_RETENTION_DAYS", None)
    os.environ.pop("WS_WAL_PRUNE_INTERVAL_HOURS", None)
