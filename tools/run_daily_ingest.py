"""Daily ingest orchestrator: run gap-fill then start WS client to resume live ingestion.

This script is intended to be invoked by cron or systemd timer. It:
- runs `tools/gap_fill.py` for a configurable lookback window (default last 7 days)
- starts `KrakenWSClient` and runs it until interrupted

Usage:
    python tools/run_daily_ingest.py --symbol XBT/USD --lookback-days 2
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta

from src.ingestion.providers.kraken_ws import KrakenWSClient
from tools.gap_fill import gap_fill

LOG = logging.getLogger(__name__)


def _run_gap_fill(out_root: str, symbol: str, lookback_days: int):
    now = datetime.utcnow()
    start = now - timedelta(days=lookback_days)
    # align to full days
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    end = now
    LOG.info("running gap fill for %s from %s to %s", symbol, start.isoformat(), end.isoformat())
    gap_fill(out_root, symbol, start, end)


async def _run_ws_client(out_root: str, symbol: str):
    client = KrakenWSClient(out_root=out_root)
    await client.start()
    LOG.info("KrakenWSClient started; press Ctrl-C to stop")
    # wait until cancelled
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await client.stop()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, required=True)
    p.add_argument("--out-root", type=str, default="data/raw")
    p.add_argument("--lookback-days", type=int, default=1, help="How many days to gap-fill before starting WS")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    try:
        _run_gap_fill(args.out_root, args.symbol, args.lookback_days)
    except Exception:
        LOG.exception("gap fill failed; proceeding to start WS client anyway")

    loop = asyncio.get_event_loop()

    ws_task = loop.create_task(_run_ws_client(args.out_root, args.symbol))

    def _on_signal(signame):
        LOG.info("received signal %s, shutting down", signame)
        ws_task.cancel()

    for s in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(s, lambda s=s: _on_signal(s))

    try:
        loop.run_until_complete(ws_task)
    except Exception:
        LOG.exception("ws client terminated with exception")
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass


if __name__ == "__main__":
    main()
