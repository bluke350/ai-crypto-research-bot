import asyncio
import os
import tempfile
from datetime import datetime, timedelta

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.ingestion.providers.kraken_ws import KrakenWSClient


@pytest.mark.asyncio
async def test_resync_pair_writes_minute_and_wal(tmp_path, monkeypatch):
    out_root = str(tmp_path)
    client = KrakenWSClient(out_root=out_root)

    pair = "XBTUSD"
    now = pd.Timestamp.utcnow().floor('s')
    # build fake trades DataFrame spanning two minutes
    rows = []
    for i in range(4):
        ts = now + pd.Timedelta(seconds=i * 15)
        rows.append({"timestamp": ts, "price": 100.0 + i, "size": 0.1 + 0.01 * i, "side": 'b'})
    trades = pd.DataFrame(rows)

    # monkeypatch kraken_rest.get_trades to return our trades
    monkeypatch.setattr('src.ingestion.providers.kraken_rest.get_trades', lambda pair_arg, since, end: trades)

    # call async resync
    await client._resync_pair(pair, int((now - pd.Timedelta(seconds=10)).timestamp()))

    # check that minute parquet files were created under data/raw
    data_dir = os.path.join(out_root, pair)
    assert os.path.exists(data_dir), "data dir should exist"
    days = os.listdir(data_dir)
    assert days, "day dir(s) should be present"

    # check wal files were written under wal_folder
    wal_dir = os.path.join(client.wal_folder, pair)
    assert os.path.exists(wal_dir), "wal dir should exist"
    found = False
    for root, dirs, files in os.walk(wal_dir):
        for f in files:
            if f.endswith('.parquet'):
                found = True
    assert found, "wal parquet files should have been written"

    # _last_ts should be updated to latest trade timestamp (seconds)
    assert pair in client._last_ts
    assert client._last_ts[pair] > 0


def _make_wal_parquet(tmp_wal_dir, pair, ts):
    # create wal folder structure and a parquet containing a single trade row
    day = ts.strftime('%Y%m%d')
    wal_pair_day = os.path.join(tmp_wal_dir, pair, day)
    os.makedirs(wal_pair_day, exist_ok=True)
    fname = ts.strftime('%Y%m%dT%H%M%S%f') + '.parquet'
    path = os.path.join(wal_pair_day, fname)
    df = pd.DataFrame([{"timestamp": pd.to_datetime(ts, utc=True), "price": 123.45, "size": 0.1}])
    pq.write_table(pa.Table.from_pandas(df), path)
    return path


@pytest.mark.asyncio
async def test_recover_wal_requeues_and_archives(tmp_path):
    out_root = str(tmp_path)
    client = KrakenWSClient(out_root=out_root)
    # create a wal parquet under client.wal_folder
    pair = 'XBTUSD'
    now = datetime.utcnow()
    p = _make_wal_parquet(client.wal_folder, pair, now)

    # ensure file exists
    assert os.path.exists(p)

    # run recovery
    client._recover_wal()

    # allow event loop to process queued callbacks
    await asyncio.sleep(0.1)

    # msg_queue should have at least one message
    # retrieve one message without blocking too long
    try:
        msg = client.msg_queue.get_nowait()
    except asyncio.QueueEmpty:
        # try awaiting briefly
        msg = await asyncio.wait_for(client.msg_queue.get(), timeout=1.0)

    assert isinstance(msg, dict)
    assert msg.get('type') == 'trade'
    assert msg.get('pair') == pair

    # original wal file should have been moved to archive
    archive_root = os.path.join(client.wal_folder, 'archive')
    assert os.path.exists(archive_root)
    archived_files = []
    for root, dirs, files in os.walk(archive_root):
        for f in files:
            if f.endswith('.parquet'):
                archived_files.append(os.path.join(root, f))
    assert archived_files, 'wal parquet should be archived after recovery'
