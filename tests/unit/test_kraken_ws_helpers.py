import asyncio
import os
import shutil
import tarfile
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.ingestion.providers.kraken_ws import KrakenWSClient


class DummyWS:
    def __init__(self):
        self._closed = False

    def __aiter__(self):
        # no messages
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def close(self):
        self._closed = True


def test_checkpoint_trades_and_flush(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))
    # ensure rate limiter no-ops in case used elsewhere
    monkeypatch.setattr(client, "_wal_flush_interval", 0.1)

    # create a trade message
    msg = {
        "type": "trade",
        "pair": "XBTUSD",
        "timestamp": int(datetime.now(timezone.utc).timestamp()),
        "price": "30000",
        "size": "0.1",
        "seq": 1,
    }

    # invoke checkpoint_trades (async)
    asyncio.run(client._checkpoint_trades(msg))

    # verify parquet minute file was written
    pair_dir = out / "XBTUSD"
    assert pair_dir.exists()
    days = list(pair_dir.iterdir())
    assert len(days) == 1
    files = list(days[0].glob("*.parquet"))
    assert len(files) == 1

    # verify wal buffer has entry
    assert any(client._wal_buffer.values())

    # flush wal synchronously (async helper)
    asyncio.run(client._flush_wal_once())

    # WAL files should have been created under wal folder
    wal_root = out / "_wal"
    assert wal_root.exists()
    # find pair dir inside wal
    wal_pair = wal_root / "XBTUSD"
    assert wal_pair.exists()
    # day dir should have parquet files
    day_dirs = list(wal_pair.iterdir())
    assert any(d.is_dir() for d in day_dirs)


def test_compress_and_prune_archived_wal(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    archive = out / "_wal" / "archive" / "XBTUSD"
    day = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y%m%d")
    day_dir = archive / day
    os.makedirs(day_dir, exist_ok=True)
    # place a dummy file
    f = day_dir / "dummy.parquet"
    with open(f, "w") as fh:
        fh.write("x")

    # set compress_after to 0 days so it will compress
    client._wal_compress_after_days = 0
    # call compress helper
    client.compress_archived_wal_once()

    # original day dir should be removed and .tar.gz present
    tar_path = str(day_dir) + ".tar.gz"
    assert os.path.exists(tar_path)
    assert not os.path.exists(day_dir)

    # set retention to 0 so prune should remove the archive
    client._wal_retention_days = 0
    # ensure mtime on tar is old
    old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).timestamp()
    os.utime(tar_path, (old_ts, old_ts))

    client.prune_archived_wal_once()
    # tar.gz should be removed
    assert not os.path.exists(tar_path)


def test_run_connect_attempts(monkeypatch):
    client = KrakenWSClient(out_root=".")
    client._backoff_test_mode = True

    calls = {"i": 0}

    async def fake_connect(*args, **kwargs):
        i = calls["i"]
        calls["i"] += 1
        if i < 2:
            raise Exception("connect fail")
        return DummyWS()

    async def run():
        # patch websockets.connect
        import websockets

        monkeypatch.setattr(websockets, "connect", fake_connect)
        attempts, success = await client.run_connect_attempts(max_attempts=4, cap_sleep=0.01)
        return attempts, success

    attempts, success = asyncio.run(run())
    # we expect success and that attempts did not exceed max_attempts
    assert success is True
    assert 1 <= attempts <= 4
