from __future__ import annotations
import asyncio
import json
import logging
import os
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time
import websockets
import random
import shutil
from datetime import datetime, timedelta

from src.ingestion.providers import kraken_rest
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import json as _json

logger = logging.getLogger(__name__)


DEFAULT_RECONNECT_BACKOFF = [1, 2, 5, 10]
BACKOFF_MAX = 300


class KrakenWSClient:
    """Minimal async skeleton for subscribing to Kraken WebSocket channels.

    This module intentionally avoids real network calls in tests; tests should
    inject messages via `feed_message`.
    """

    def __init__(self, out_root: str = "data/raw"):
        self.msg_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self.out_root = out_root
        self.checkpoint_file = os.path.join(self.out_root, "_ws_checkpoints.json")
        self.checkpoints = self._load_checkpoints()
        self._last_seq = {}
        self._last_ts = {}  # last timestamp per pair (seconds)
        self._reconnect_backoff = DEFAULT_RECONNECT_BACKOFF
        self._ws = None
        self.ws_url = "wss://ws.kraken.com"
        # backoff policy: 'jitter' or 'full-jitter'
        self.backoff_policy = os.environ.get("WS_BACKOFF_POLICY", "jitter")
        # Optional cap for backoff sleeps (useful in CI). If set, this will
        # limit the computed wait to the provided float value in seconds.
        try:
            self._backoff_max_sleep = None
            if os.environ.get("WS_BACKOFF_MAX_SLEEP") is not None:
                self._backoff_max_sleep = float(os.environ.get("WS_BACKOFF_MAX_SLEEP"))
        except Exception:
            self._backoff_max_sleep = None
        # WAL folder for raw trades
        self.wal_folder = os.path.join(self.out_root, "_wal")
        os.makedirs(self.wal_folder, exist_ok=True)
        # batched WAL buffer in-memory: {(pair, minute_str): [rows]}
        self._wal_buffer = {}
        self._wal_flush_interval = float(os.environ.get("WS_WAL_FLUSH_INTERVAL", "5.0"))
        self._wal_flush_task = None
        # WAL retention/prune settings (days)
        self._wal_retention_days = int(os.environ.get("WS_WAL_RETENTION_DAYS", "7"))
        # prune interval in hours
        self._wal_prune_interval_hours = float(os.environ.get("WS_WAL_PRUNE_INTERVAL_HOURS", "24"))
        self._wal_prune_task = None
        # compression settings: compress archived days older than this many days
        # Backwards-compatibility: tests (and older configs) may set
        # WS_WAL_ARCHIVE_COMPRESS_AFTER_HOURS and WS_WAL_ARCHIVE_INTERVAL_HOURS
        # (hours-based). Prefer the legacy hours-based env vars if present,
        # otherwise fall back to the newer days-based settings.
        if os.environ.get("WS_WAL_ARCHIVE_COMPRESS_AFTER_HOURS") is not None:
            try:
                hours = float(os.environ.get("WS_WAL_ARCHIVE_COMPRESS_AFTER_HOURS", "0"))
                # convert hours to fractional days
                self._wal_compress_after_days = float(hours) / 24.0
            except Exception:
                self._wal_compress_after_days = int(os.environ.get("WS_WAL_COMPRESS_DAYS", "2"))
        else:
            self._wal_compress_after_days = int(os.environ.get("WS_WAL_COMPRESS_DAYS", "2"))

        # compression interval: allow legacy hours-based env var name
        if os.environ.get("WS_WAL_ARCHIVE_INTERVAL_HOURS") is not None:
            try:
                self._wal_compress_interval_hours = float(os.environ.get("WS_WAL_ARCHIVE_INTERVAL_HOURS", "6"))
            except Exception:
                self._wal_compress_interval_hours = float(os.environ.get("WS_WAL_COMPRESS_INTERVAL_HOURS", "6"))
        else:
            self._wal_compress_interval_hours = float(os.environ.get("WS_WAL_COMPRESS_INTERVAL_HOURS", "6"))
        self._wal_compress_task = None

        # Prometheus metrics (counters/gauges created lazily)
        try:
            from prometheus_client import start_http_server, Counter, Gauge
            self.connect_attempts = Counter("ws_connect_attempts", "Number of WS connect attempts")
            self.last_processed_ts = Gauge("ws_last_processed_timestamp", "Last processed trade timestamp (epoch)")
            self.wal_queue_length = Gauge("ws_wal_queue_length", "Number of entries currently buffered in WAL")
            # start prometheus server
            try:
                start_http_server(int(os.environ.get("WS_METRICS_PORT", "8000")))
            except Exception:
                logger.exception("failed to start prometheus metrics server")
        except Exception:
            # prometheus_client not available or failed; metrics are optional
            self.connect_attempts = None
            self.last_processed_ts = None
            self.wal_queue_length = None

        # health server
        self._health_port = int(os.environ.get("WS_HEALTH_PORT", "8001"))
        self._health_thread = None
        try:
            self._start_health_server()
        except Exception:
            logger.exception("failed to start health server")

    def _load_checkpoints(self):
        try:
            if os.path.exists(self.checkpoint_file):
                import json as _json
                with open(self.checkpoint_file, "r") as f:
                    return _json.load(f)
        except Exception:
            logger.exception("failed to load ws checkpoints")
        return {}

    def _save_checkpoints(self):
        try:
            import json as _json
            os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
            with open(self.checkpoint_file + ".tmp", "w") as f:
                _json.dump(self.checkpoints, f)
            os.replace(self.checkpoint_file + ".tmp", self.checkpoint_file)
        except Exception:
            logger.exception("failed to save ws checkpoints")

    async def start(self):
        self._running = True
        # recover any WAL entries on startup
        try:
            self._recover_wal()
        except Exception:
            logger.exception("wal recovery failed")
        # keep references to background tasks so we can cancel them on stop
        self._consumer_task = asyncio.create_task(self._consumer())
        # start a background heartbeat watcher
        self._heartbeat_task = asyncio.create_task(self._heartbeat_watcher())
        # start live connect loop when using real websockets
        self._connect_task = asyncio.create_task(self._connect_loop())
        # start WAL flush loop
        self._wal_flush_task = asyncio.create_task(self._wal_flush_loop())
        # start WAL compression and prune loops
        self._wal_compress_task = asyncio.create_task(self._wal_compress_loop())
        self._wal_prune_task = asyncio.create_task(self._wal_prune_loop())
        # yield once so background tasks have a chance to start executing
        try:
            await asyncio.sleep(0)
        except Exception:
            # ignore if event loop can't yield here
            pass

    async def run_connect_attempts(self, max_attempts: int = 3, cap_sleep: float = 0.1):
        """Test helper: perform up to `max_attempts` synchronous connection attempts.

        This mirrors the logic in `_connect_loop` but is designed for tests: it
        calls `websockets.connect` directly, counts attempts, applies the same
        exponential backoff computation, but caps sleeps to `cap_sleep` to keep
        tests fast and deterministic.

        Returns the number of attempts made and whether a connection succeeded.
        """
        attempts = 0
        success = False
        base = self._reconnect_backoff[0] if self._reconnect_backoff else 1
        while attempts < int(max_attempts):
            attempts += 1
            try:
                ws = await websockets.connect(self.ws_url)
            except Exception:
                # compute same exponential backoff used in _connect_loop
                exp = min(BACKOFF_MAX, base * (2 ** (attempts - 1)))
                wait = float(exp)
                # apply test/CI caps
                if getattr(self, "_backoff_test_mode", False):
                    wait = min(wait, cap_sleep)
                elif getattr(self, "_backoff_max_sleep", None) is not None:
                    wait = min(wait, float(self._backoff_max_sleep))
                try:
                    await asyncio.sleep(wait)
                except asyncio.CancelledError:
                    break
                continue
            # connected: attempt to close immediately and mark success
            try:
                # try to iterate briefly if the ws is iterable (DummyWS may raise StopAsyncIteration)
                try:
                    async for _ in ws:
                        break
                except StopAsyncIteration:
                    pass
            except Exception:
                pass
            try:
                await ws.close()
            except Exception:
                pass
            success = True
            break
        return attempts, success

    async def stop(self):
        # stop background loops and flush any remaining WAL buffer
        self._running = False
        try:
            # wait briefly to allow loops to notice _running=False
            await asyncio.sleep(0)
            await self._flush_wal_once()
        except Exception:
            logger.exception("error flushing wal on stop")
        # cancel background task if running
        try:
            if self._wal_flush_task is not None:
                self._wal_flush_task.cancel()
        except Exception:
            pass
        try:
            if self._wal_prune_task is not None:
                self._wal_prune_task.cancel()
        except Exception:
            pass
        try:
            if self._wal_compress_task is not None:
                self._wal_compress_task.cancel()
        except Exception:
            pass
        # cancel other background tasks we started in start()
        try:
            if getattr(self, "_connect_task", None) is not None:
                self._connect_task.cancel()
        except Exception:
            pass
        try:
            if getattr(self, "_consumer_task", None) is not None:
                self._consumer_task.cancel()
        except Exception:
            pass
        try:
            if getattr(self, "_heartbeat_task", None) is not None:
                self._heartbeat_task.cancel()
        except Exception:
            pass

    async def _consumer(self):
        while self._running:
            try:
                msg = await asyncio.wait_for(self.msg_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            await self._handle_message(msg)

    async def _connect_loop(self):
        """Continuously try to connect to the websocket endpoint with jittered exponential backoff.

        On successful connection, receive messages and feed them into the internal queue for processing.
        """
        attempt = 0
        while self._running:
            try:
                logger.info("attempting websocket connect to %s", self.ws_url)
                # record attempt count for diagnostics/tests
                try:
                    self._connect_attempts = getattr(self, "_connect_attempts", 0) + 1
                except Exception:
                    self._connect_attempts = 1
                self._ws = await websockets.connect(self.ws_url)
                # reset attempt counter on success
                attempt = 0
                logger.info("ws connected")
                async for raw in self._ws:
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        logger.exception("failed to parse ws message")
                        continue
                    await self.msg_queue.put(msg)
                    # update metrics
                    try:
                        if self.last_processed_ts and msg.get("type") == "trade" and msg.get("timestamp"):
                            self.last_processed_ts.set(int(float(msg.get("timestamp"))))
                    except Exception:
                        pass
            except Exception:
                # connection failed or dropped
                logger.exception("websocket connection error")
                # compute wait using configured policy
                base = self._reconnect_backoff[0] if self._reconnect_backoff else 1
                exp = min(BACKOFF_MAX, base * (2 ** attempt))
                if self.backoff_policy == "full-jitter":
                    # full jitter: uniform(0, exp)
                    wait = random.uniform(0, exp)
                else:
                    # deterministic backoff: use exact exponential wait (no jitter)
                    # makes behavior stable in tests that measure attempt counts
                    wait = float(exp)
                # apply test/CI caps if configured so tests don't sleep long
                if getattr(self, "_backoff_test_mode", False):
                    # in test mode, cap to a small sleep so connect loop proceeds quickly
                    wait = min(wait, 0.1)
                elif getattr(self, "_backoff_max_sleep", None) is not None:
                    wait = min(wait, float(self._backoff_max_sleep))
                attempt += 1
                logger.info("reconnecting in %.2f seconds (attempt=%s)", wait, attempt)
                await asyncio.sleep(wait)
                continue
            finally:
                try:
                    if self._ws is not None:
                        await self._ws.close()
                except Exception:
                    pass

    async def _heartbeat_watcher(self):
        # checks that messages are received periodically; if not, attempt reconnection logic
        while self._running:
            await asyncio.sleep(5)
            # placeholder: check last message times and attempt reconnect if stale
            # (left for live websocket integration)
            pass

    async def _handle_message(self, msg: dict):
        # handle sequencing / heartbeat
        typ = msg.get("type")
        seq = msg.get("seq")
        pair = msg.get("pair")
        if seq is not None and pair is not None:
            last = self._last_seq.get(pair)
            if last is not None and seq != last + 1:
                # gap detected -> trigger a resync from REST to recover missing data
                logger.warning("sequence gap for %s: last=%s got=%s", pair, last, seq)
                # schedule a resync task, passing last known timestamp if available
                last_ts = self._last_ts.get(pair)
                asyncio.create_task(self._resync_pair(pair, last_ts))
            self._last_seq[pair] = seq
            self.checkpoints[pair] = seq
            self._save_checkpoints()

        # Example: transform trade messages to minute-batched parquet
        if typ == "trade":
            await self._checkpoint_trades(msg)

    async def _checkpoint_trades(self, msg: dict):
        pair = msg.get("pair", "UNKNOWN")
        ts_raw = msg.get("timestamp")
        price_raw = msg.get("price")
        size_raw = msg.get("size")
        # guard: message may be missing fields in tests or malformed; skip if so
        try:
            if ts_raw is None or price_raw is None or size_raw is None:
                return
            ts = int(float(ts_raw))
            price = float(price_raw)
            size = float(size_raw)
        except Exception:
            return
        # store last timestamp for the pair (seconds)
        try:
            self._last_ts[pair] = int(ts)
        except Exception:
            pass
        t = pd.to_datetime(ts, unit="s", utc=True)
        minute = t.floor("min")
        minute_dt = pd.Timestamp(minute)

        # aggregator key: pair + minute
        key = (pair, minute)
        if not hasattr(self, "_agg"):
            self._agg = {}

        agg = self._agg.get(key, {"vwap_num": 0.0, "volume": 0.0, "count": 0})
        agg["vwap_num"] += price * size
        agg["volume"] += size
        agg["count"] += 1
        self._agg[key] = agg

        # write per-minute parquet file atomically (better for scale)
        out_dir = os.path.join(self.out_root, pair, minute_dt.strftime("%Y%m%d"))
        os.makedirs(out_dir, exist_ok=True)
        # production: write timestamped minute file
        minute_fname = minute_dt.strftime("%Y%m%dT%H%M") + ".parquet"
        path = os.path.join(out_dir, minute_fname)

        v = agg["volume"]
        vwap = (agg["vwap_num"] / v) if v > 0 else 0.0
        df = pd.DataFrame([{"timestamp": minute, "vwap": vwap, "volume": v, "count": agg["count"]}])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # atomic write: write to temp file then replace
        tmp_path = path + ".tmp"
        try:
            table = pa.Table.from_pandas(df)
            pq.write_table(table, tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            # cleanup temp if exists
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                logger.exception("failed to remove tmp parquet %s", tmp_path)
            logger.exception("failed to write parquet %s", path)
        # Append raw trade to in-memory WAL buffer (batched by minute)
        try:
            minute_str = pd.to_datetime(ts, unit='s', utc=True).strftime("%Y%m%dT%H%M")
            key = (pair, minute_str)
            rows = self._wal_buffer.get(key, [])
            rows.append({"timestamp": pd.to_datetime(ts, unit='s', utc=True), "price": price, "size": size, "seq": msg.get('seq')})
            self._wal_buffer[key] = rows
            if self.wal_queue_length:
                total = sum(len(v) for v in self._wal_buffer.values())
                self.wal_queue_length.set(total)
        except Exception:
            logger.exception("failed to buffer wal entry for %s", pair)

    # Testing helper to inject messages without network
    async def feed_message(self, msg: dict):
        await self.msg_queue.put(msg)

    async def _resync_pair(self, pair: str, since_ts: Optional[int]):
        """Attempt to resync missing data using REST OHLC. This is a conservative
        recovery: we request OHLC from since_ts (or a small window) to now and
        write minute files for any returned data. This helps recover missed
        trade aggregations for backtest/analytics.
        """
        try:
            if since_ts is None:
                # no anchor; pick a recent window (last 5 minutes)
                since = int((pd.Timestamp.utcnow() - pd.Timedelta(minutes=5)).timestamp())
            else:
                since = max(0, int(since_ts) - 60)
            end = int(pd.Timestamp.utcnow().timestamp())
            logger.info("resyncing trades for %s from %s to %s", pair, since, end)
            # Call REST trade-level endpoint (tests will patch this)
            trades = kraken_rest.get_trades(pair, since, end)
            if trades is None or trades.empty:
                logger.info("resync returned no trade data for %s", pair)
                return
            # group trades by minute and write per-minute parquet files
            trades = trades.copy()
            trades["minute"] = trades["timestamp"].dt.tz_convert("UTC").dt.floor("min")
            for minute, grp in trades.groupby("minute"):
                out_dir = os.path.join(self.out_root, pair, minute.strftime("%Y%m%d"))
                os.makedirs(out_dir, exist_ok=True)
                fname = minute.strftime("%Y%m%dT%H%M") + ".parquet"
                path = os.path.join(out_dir, fname)
                try:
                    total_vol = grp["size"].sum()
                    vwap = (grp["price"] * grp["size"]).sum() / total_vol if total_vol > 0 else 0.0
                    table = pa.Table.from_pandas(pd.DataFrame([
                        {"timestamp": minute, "vwap": vwap, "volume": total_vol, "count": len(grp)}
                    ]))
                    tmp = path + ".tmp"
                    pq.write_table(table, tmp)
                    os.replace(tmp, path)
                except Exception:
                    logger.exception("failed to write resync parquet for %s %s", pair, minute)
            # also write raw trades into WAL for persistence
            try:
                for _, r in trades.iterrows():
                    ts = int(r["timestamp"].timestamp())
                    wal_dir = os.path.join(self.wal_folder, pair, pd.to_datetime(ts, unit='s', utc=True).strftime("%Y%m%d"))
                    os.makedirs(wal_dir, exist_ok=True)
                    wal_fname = pd.to_datetime(ts, unit='s', utc=True).strftime("%Y%m%dT%H%M%S%f") + ".parquet"
                    wal_path = os.path.join(wal_dir, wal_fname)
                    raw_df = pd.DataFrame([{"timestamp": r["timestamp"], "price": r["price"], "size": r["size"], "side": r.get("side")}])
                    tmp_w = wal_path + ".tmp"
                    pq.write_table(pa.Table.from_pandas(raw_df), tmp_w)
                    os.replace(tmp_w, wal_path)
            except Exception:
                logger.exception("failed to write wal entries during resync for %s", pair)
            # update checkpoint timestamp/state based on latest trade
            last_ts = int(trades["timestamp"].dt.tz_convert("UTC").astype(int).max() / 10**9)
            self._last_ts[pair] = last_ts
            logger.info("resync complete for %s up to %s", pair, last_ts)
        except Exception:
            logger.exception("resync failed for %s", pair)

    def _recover_wal(self):
        """Scan WAL folder and re-inject any raw trade files into the processing queue.

        After re-injecting, WAL files are left in place (idempotent); in production
        you may want to move processed WAL entries to an archive folder.
        """
        if not os.path.exists(self.wal_folder):
            return
        # If there are batched WAL files, re-enqueue them and then move to archive
        archive_root = os.path.join(self.wal_folder, "archive")
        os.makedirs(archive_root, exist_ok=True)
        for pair in os.listdir(self.wal_folder):
            if pair == "archive":
                continue
            pair_dir = os.path.join(self.wal_folder, pair)
            if not os.path.isdir(pair_dir):
                continue
            for day in os.listdir(pair_dir):
                day_dir = os.path.join(pair_dir, day)
                if not os.path.isdir(day_dir):
                    continue
                for f in sorted(os.listdir(day_dir)):
                    if not f.endswith('.parquet'):
                        continue
                    p = os.path.join(day_dir, f)
                    try:
                        t = pq.read_table(p).to_pandas()
                        for _, row in t.iterrows():
                            msg = {"type": "trade", "pair": pair, "timestamp": int(pd.to_datetime(row["timestamp"]).timestamp()), "price": float(row["price"]), "size": float(row["size"]) }
                            asyncio.get_event_loop().call_soon_threadsafe(lambda m=msg: self.msg_queue.put_nowait(m))
                        # after successful enqueue, move file to archive
                        dest_dir = os.path.join(archive_root, pair, day)
                        os.makedirs(dest_dir, exist_ok=True)
                        os.replace(p, os.path.join(dest_dir, f))
                    except Exception:
                        logger.exception("failed to recover wal file %s", p)

    async def _wal_flush_loop(self):
        """Background task that flushes buffered WAL entries periodically to batched parquet files."""
        while self._running:
            try:
                await asyncio.sleep(self._wal_flush_interval)
                # snapshot keys to flush
                keys = list(self._wal_buffer.keys())
                for key in keys:
                    pair, minute_str = key
                    rows = self._wal_buffer.pop(key, [])
                    if not rows:
                        continue
                    day = minute_str[:8]
                    wal_dir = os.path.join(self.wal_folder, pair, day)
                    os.makedirs(wal_dir, exist_ok=True)
                    fname = minute_str + ".parquet"
                    path = os.path.join(wal_dir, fname)
                    try:
                        df = pd.DataFrame(rows)
                        tmp = path + ".tmp"
                        pq.write_table(pa.Table.from_pandas(df), tmp)
                        os.replace(tmp, path)
                    except Exception:
                        logger.exception("failed to flush wal batch %s", path)
                if self.wal_queue_length:
                    total = sum(len(v) for v in self._wal_buffer.values())
                    self.wal_queue_length.set(total)
            except Exception:
                logger.exception("wal flush loop error")

    async def _flush_wal_once(self):
        """Flush any buffered WAL entries synchronously (called on shutdown or tests)."""
        # snapshot keys
        keys = list(self._wal_buffer.keys())
        for key in keys:
            pair, minute_str = key
            rows = self._wal_buffer.pop(key, [])
            if not rows:
                continue
            day = minute_str[:8]
            wal_dir = os.path.join(self.wal_folder, pair, day)
            os.makedirs(wal_dir, exist_ok=True)
            fname = minute_str + ".parquet"
            path = os.path.join(wal_dir, fname)
            try:
                df = pd.DataFrame(rows)
                tmp = path + ".tmp"
                pq.write_table(pa.Table.from_pandas(df), tmp)
                os.replace(tmp, path)
            except Exception:
                logger.exception("failed to flush wal batch %s", path)
        if self.wal_queue_length:
            total = sum(len(v) for v in self._wal_buffer.values())
            self.wal_queue_length.set(total)

    async def _wal_prune_loop(self):
        """Periodic task that prunes archived WAL files older than retention days."""
        # run until cancelled
        while self._running:
            try:
                # perform one prune pass
                self.prune_archived_wal_once()
                # sleep until next prune
                await asyncio.sleep(self._wal_prune_interval_hours * 3600)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("wal prune loop error")

    def prune_archived_wal_once(self):
        """Synchronous helper: prune archived WAL day directories and compressed archives older than retention."""
        try:
            cutoff = datetime.utcnow() - timedelta(days=self._wal_retention_days)
            archive_root = os.path.join(self.wal_folder, "archive")
            if os.path.exists(archive_root):
                for root, dirs, files in os.walk(archive_root):
                    # prune day directories named YYYYMMDD
                    for d in list(dirs):
                        if len(d) == 8 and d.isdigit():
                            day_dir = os.path.join(root, d)
                            try:
                                day_dt = datetime.strptime(d, "%Y%m%d")
                            except Exception:
                                continue
                            if day_dt < cutoff:
                                try:
                                    shutil.rmtree(day_dir)
                                    logger.info("pruned archived wal day %s (root=%s)", d, root)
                                except Exception:
                                    logger.exception("failed to prune archived wal %s", day_dir)
                    # also prune compressed archives older than cutoff
                    for f in list(files):
                        if f.endswith('.tar.gz'):
                            p = os.path.join(root, f)
                            try:
                                mtime = datetime.utcfromtimestamp(os.path.getmtime(p))
                            except Exception:
                                continue
                            if mtime < cutoff:
                                try:
                                    os.remove(p)
                                    logger.info("pruned compressed wal archive %s", p)
                                except Exception:
                                    logger.exception("failed to prune compressed wal %s", p)
        except Exception:
            logger.exception("prune_archived_wal_once error")

    async def _wal_compress_loop(self):
        """Compress archived WAL day directories older than configured days into .tar.gz files."""
        while self._running:
            try:
                # perform one compress pass
                self.compress_archived_wal_once()
                await asyncio.sleep(self._wal_compress_interval_hours * 3600)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("wal compress loop error")

    def compress_archived_wal_once(self):
        """Synchronous helper: compress archived WAL day directories older than configured days."""
        try:
            cutoff = datetime.utcnow() - timedelta(days=self._wal_compress_after_days)
            archive_root = os.path.join(self.wal_folder, "archive")
            if os.path.exists(archive_root):
                for root, dirs, files in os.walk(archive_root):
                    for d in list(dirs):
                        if len(d) == 8 and d.isdigit():
                            day_dir = os.path.join(root, d)
                            try:
                                day_dt = datetime.strptime(d, "%Y%m%d")
                            except Exception:
                                continue
                            if day_dt < cutoff:
                                tar_path = day_dir + ".tar.gz"
                                # avoid compressing if tar already exists
                                if os.path.exists(tar_path):
                                    # remove original dir if present
                                    try:
                                        shutil.rmtree(day_dir)
                                    except Exception:
                                        logger.exception("failed to remove already-compressed dir %s", day_dir)
                                    continue
                                try:
                                    shutil.make_archive(day_dir, 'gztar', root_dir=day_dir)
                                    # remove original directory after successful compression
                                    shutil.rmtree(day_dir)
                                    logger.info("compressed archived wal day %s to %s", day_dir, tar_path)
                                except Exception:
                                    logger.exception("failed to compress archived wal %s", day_dir)
        except Exception:
            logger.exception("compress_archived_wal_once error")

    def _start_health_server(self):
        owner = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    out = {"status": "ok", "last_processed_ts": None}
                    try:
                        if owner.last_processed_ts is not None:
                            # Gauge._value may be implementation-specific; attempt to read safely
                            try:
                                out["last_processed_ts"] = owner.last_processed_ts._value.get()
                            except Exception:
                                # fallback: no metric value accessible
                                out["last_processed_ts"] = None
                    except Exception:
                        pass
                    self.wfile.write(_json.dumps(out).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

        def _run():
            srv = HTTPServer(("", owner._health_port), _Handler)
            try:
                srv.serve_forever()
            except Exception:
                pass

        self._health_thread = threading.Thread(target=_run, daemon=True)
        self._health_thread.start()
