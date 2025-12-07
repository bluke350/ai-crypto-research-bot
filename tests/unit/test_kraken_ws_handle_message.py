import asyncio
import os
import pandas as pd

from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_handle_message_missing_fields_does_not_crash(tmp_path):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    # message missing seq/pair should be ignored gracefully
    msg = {"type": "trade", "timestamp": 1234567890, "price": 100.0, "size": 0.1}
    # should not raise
    asyncio.run(client._handle_message(msg))


def test_handle_message_seq_gap_triggers_resync_but_no_data(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    pair = "XBTUSD"
    client._last_seq[pair] = 1

    # patch the kraken_rest used by kraken_ws to return an empty DataFrame
    import src.ingestion.providers.kraken_ws as kw

    monkeypatch.setattr(kw.kraken_rest, "get_trades", lambda p, s, e: pd.DataFrame())

    created = []
    orig_create = asyncio.create_task

    def create_task(coro):
        t = orig_create(coro)
        created.append(t)
        return t

    monkeypatch.setattr(asyncio, "create_task", create_task)

    msg = {"type": "trade", "pair": pair, "seq": 3}
    asyncio.run(client._handle_message(msg))

    # resync task scheduled
    if created:
        assert created[0].done()

    # no parquet files should be created since resync returned no data
    assert not (out / pair).exists()
