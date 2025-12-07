import asyncio
import os
import pytest
from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_resync_with_no_trades_returns(tmp_path, monkeypatch):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))
    # patch get_trades to return None
    monkeypatch.setattr('src.ingestion.providers.kraken_ws.kraken_rest', type('ns', (), {'get_trades': lambda p, s, e: None}))
    # call resync synchronously and ensure no exception
    asyncio.run(client._resync_pair('XBT/USD', None))
    # no trade parquet nor wal should be created
    pair_dir = tmp_path / 'data' / 'XBT/USD'
    assert not pair_dir.exists()
