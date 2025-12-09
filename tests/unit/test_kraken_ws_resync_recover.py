import asyncio
import os
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_handle_message_triggers_resync_and_writes(tmp_path, monkeypatch):
    out = tmp_path / "data"
    # create client after we prepare the monkeypatch target so the module-level
    # reference to `kraken_rest` used by `kraken_ws` is the same object we
    # patch below.
    client = KrakenWSClient(out_root=str(out))

    pair = "XBTUSD"
    # set last_seq so a gap will be detected (last=1, incoming seq=3)
    client._last_seq[pair] = 1

    # prepare fake trades DataFrame returned by kraken_rest.get_trades
    ts = pd.to_datetime(datetime.utcnow()).tz_localize("UTC")
    trades = pd.DataFrame([{"timestamp": ts, "price": 30000.0, "size": 0.1}])

    # patch the kraken_rest used by the kraken_ws module so `_resync_pair`
    # calls our fake get_trades implementation (avoids calling real requests)
    import src.ingestion.providers.kraken_ws as kw

    monkeypatch.setattr(kw.kraken_rest, "get_trades", lambda p, s, e: trades)

    # capture created asyncio tasks so we can await the resync task
    created = []
    orig_create = asyncio.create_task

    def create_task(coro):
        t = orig_create(coro)
        created.append(t)
        return t

    monkeypatch.setattr(asyncio, "create_task", create_task)

    msg = {"type": "trade", "pair": pair, "seq": 3}

    # run the _handle_message coroutine which schedules the resync task
    asyncio.run(client._handle_message(msg))


    # the resync task should have been scheduled and (in this test) already run
    if created:
        assert created[0].done()

    # verify parquet files written for resync
    pair_dir = out / pair
    assert pair_dir.exists()
    # wal folder should contain parquet raw files after resync
    wal_root = out / "_wal" / pair
    assert wal_root.exists()
    # checkpoints update
    assert pair in client._last_ts


def test_recover_wal_enqueues_messages_and_moves_to_archive(tmp_path):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    pair = "XBTUSD"
    day = datetime.utcnow().strftime("%Y%m%d")
    wal_dir = os.path.join(client.wal_folder, pair, day)
    os.makedirs(wal_dir, exist_ok=True)

    # write a small parquet file with timestamp/price/size
    df = pd.DataFrame([{"timestamp": pd.to_datetime(datetime.utcnow()), "price": 30000.0, "size": 0.1}])
    path = os.path.join(wal_dir, "sample.parquet")
    pq.write_table(pa.Table.from_pandas(df), path)

    # monkeypatch asyncio.get_event_loop to provide a loop that executes call_soon_threadsafe inline
    class FakeLoop:
        def call_soon_threadsafe(self, cb):
            # call the callback immediately
            cb()

    import asyncio as _asyncio

    orig_get_loop = _asyncio.get_event_loop
    try:
        _asyncio.get_event_loop = lambda: FakeLoop()
        # run recover (synchronous)
        client._recover_wal()
    finally:
        _asyncio.get_event_loop = orig_get_loop

    # after recover, the parquet should have been moved to archive
    archive_dir = os.path.join(client.wal_folder, "archive", pair, day)
    assert os.path.exists(archive_dir)
    # message should be enqueued in client.msg_queue (use get_nowait)
    item = client.msg_queue.get_nowait()
    assert item["type"] == "trade"
    assert item["pair"] == pair
