from __future__ import annotations
import asyncio
import json
import os
from typing import Callable, Awaitable
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class KrakenWSClient:
    """Minimal async skeleton for subscribing to Kraken WebSocket channels.

    This module intentionally avoids real network calls in tests; tests should
    inject messages via `feed_message`.
    """

    def __init__(self, out_root: str = "data/raw"):
        self.msg_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self.out_root = out_root

    async def start(self):
        self._running = True
        asyncio.create_task(self._consumer())

    async def stop(self):
        self._running = False

    async def _consumer(self):
        while self._running:
            try:
                msg = await asyncio.wait_for(self.msg_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            await self._handle_message(msg)

    async def _handle_message(self, msg: dict):
        # Example: transform trade messages to minute-batched parquet (very small stub)
        typ = msg.get("type")
        if typ == "trade":
            await self._checkpoint_trades(msg)

    async def _checkpoint_trades(self, msg: dict):
        pair = msg.get("pair", "UNKNOWN")
        ts = msg.get("timestamp")
        price = msg.get("price")
        size = msg.get("size")
        out_dir = os.path.join(self.out_root, pair, pd.to_datetime(ts, unit="s").strftime("%Y%m%d"))
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "trades.parquet")
        df = pd.DataFrame([{"timestamp": pd.to_datetime(ts, unit="s", utc=True), "price": price, "size": size}])
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path)

    # Testing helper to inject messages without network
    async def feed_message(self, msg: dict):
        await self.msg_queue.put(msg)
