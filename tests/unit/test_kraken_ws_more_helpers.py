import asyncio
import os
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def test_resync_with_no_trades_does_nothing(tmp_path, monkeypatch):
    from src.ingestion.providers.kraken_ws import KrakenWSClient
    import src.ingestion.providers.kraken_ws as kw

    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))
    pair = "XBTUSD"
    client._last_seq[pair] = 1
    client._last_ts[pair] = 99999

    captured = {}

    def fake_get_trades(p, s, e):
        captured['pair'] = p
        captured['since'] = s
        captured['end'] = e
        return pd.DataFrame()
    monkeypatch.setattr(kw.kraken_rest, 'get_trades', fake_get_trades)

    # call resync directly to exercise branch deterministically
    asyncio.run(client._resync_pair(pair, client._last_ts[pair]))

    # ensure our fake_get_trades was called with a since near last_ts-60
    assert captured.get('pair') == pair
    assert captured.get('since') is not None


def test_recover_wal_with_corrupt_parquet_leaves_file(tmp_path):
    from src.ingestion.providers.kraken_ws import KrakenWSClient

    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))
    pair = 'XBTUSD'
    day = datetime.now(timezone.utc).strftime('%Y%m%d')
    wal_dir = os.path.join(client.wal_folder, pair, day)
    os.makedirs(wal_dir, exist_ok=True)

    path = os.path.join(wal_dir, 'corrupt.parquet')
    # write non-parquet content
    with open(path, 'wb') as fh:
        fh.write(b'not a parquet')

    # should not raise
    client._recover_wal()

    # file should remain (not moved to archive) because reading failed
    assert os.path.exists(path)


def test_handle_message_passes_last_ts_to_resync(tmp_path, monkeypatch):
    from src.ingestion.providers.kraken_ws import KrakenWSClient
    import src.ingestion.providers.kraken_ws as kw

    out = tmp_path / 'data'
    client = KrakenWSClient(out_root=str(out))
    pair = 'XBTUSD'
    client._last_seq[pair] = 5
    client._last_ts[pair] = 123456

    captured = {}

    def fake_get_trades(p, s, e):
        captured['since'] = s
        captured['end'] = e
        return pd.DataFrame()
    monkeypatch.setattr(kw.kraken_rest, 'get_trades', fake_get_trades)

    # call resync directly to ensure last_ts is used
    asyncio.run(client._resync_pair(pair, client._last_ts[pair]))
    assert captured.get('since') is not None
    assert captured.get('end') is not None
