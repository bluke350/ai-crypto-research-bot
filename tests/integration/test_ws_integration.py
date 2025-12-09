import asyncio
import os
import json
import pytest
import socket
import pyarrow.parquet as pq
import pandas as pd
from websockets import serve

from src.ingestion.providers.kraken_ws import KrakenWSClient


async def _run_server_and_client(tmp_path):
    # find a free port
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()

    async def handler(websocket, path=None):
        # send two trade messages then close
        msgs = [
            {"type": "trade", "pair": "XBT/USD", "timestamp": 1609459200, "price": 30000.0, "size": 0.1, "seq": 1},
            {"type": "trade", "pair": "XBT/USD", "timestamp": 1609459260, "price": 30010.0, "size": 0.2, "seq": 2},
        ]
        for m in msgs:
            await websocket.send(json.dumps(m))
        await asyncio.sleep(0.1)

    server = await serve(handler, "localhost", port)

    client = KrakenWSClient(out_root=str(tmp_path / "data"))
    client.ws_url = f"ws://localhost:{port}"

    await client.start()
    # allow server and client to exchange messages
    await asyncio.sleep(1.0)
    await client.stop()

    server.close()
    await server.wait_closed()

    # ensure parquet files exist
    day_dir = tmp_path / "data" / "XBT/USD" / pd.to_datetime(1609459200, unit='s').strftime("%Y%m%d")
    files = os.listdir(day_dir)
    parquets = [f for f in files if f.endswith('.parquet')]
    assert parquets
    # read first parquet
    df = pq.read_table(os.path.join(day_dir, parquets[0])).to_pandas()
    assert 'vwap' in df.columns


@pytest.mark.asyncio
async def test_ws_integration(tmp_path):
    await _run_server_and_client(tmp_path)
