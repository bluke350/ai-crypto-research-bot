import asyncio
import os
import tarfile
from datetime import datetime, timedelta
import pytest
from src.ingestion.providers.kraken_ws import KrakenWSClient


@pytest.mark.asyncio
async def test_wal_archiver_creates_tar_and_removes_dir(tmp_path):
    out = tmp_path / "data"
    # enable archiver
    os.environ["WS_WAL_ARCHIVE_COMPRESSION"] = "1"
    # compress any day older than 0 hours
    os.environ["WS_WAL_ARCHIVE_COMPRESS_AFTER_HOURS"] = "0"
    os.environ["WS_WAL_ARCHIVE_INTERVAL_HOURS"] = "0.001"

    client = KrakenWSClient(out_root=str(out))
    # create an archived day dir
    archive_day = datetime.utcnow().strftime("%Y%m%d")
    day_dir = os.path.join(str(out), "_wal", "archive", "XBT/USD", archive_day)
    os.makedirs(day_dir, exist_ok=True)
    with open(os.path.join(day_dir, "f.parquet"), "w") as f:
        f.write("x")

    await client.start()
    await asyncio.sleep(1.5)
    await client.stop()

    # original dir should be removed and a tar.gz should exist
    tar_path = day_dir + ".tar.gz"
    assert not os.path.exists(day_dir)
    assert os.path.exists(tar_path)
    # tar should contain the parquet file
    with tarfile.open(tar_path, "r:gz") as tar:
        names = tar.getnames()
        assert any("f.parquet" in n for n in names)

    # cleanup env
    os.environ.pop("WS_WAL_ARCHIVE_COMPRESSION", None)
    os.environ.pop("WS_WAL_ARCHIVE_COMPRESS_AFTER_HOURS", None)
    os.environ.pop("WS_WAL_ARCHIVE_INTERVAL_HOURS", None)
