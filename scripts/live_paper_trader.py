"""Live paper-trader manager.

This script launches one `tooling.run_paper_live` subprocess per pair (broker live-ws mode)
and monitors the resulting `experiments/artifacts/<run_id>/exec_log.parquet` files.

It aggregates executed fills into a small SQLite DB at `experiments/live_trades.db`
and prints a concise live summary (position, cash, estimated portfolio value) for each
managed pair.

Usage:
    python -m scripts.live_paper_trader --pairs "XBT/USD" "ETH/USD" --cash 10000

Notes:
 - Requires `websockets` and network connectivity to Kraken if using `--live-ws`.
 - This manager is intentionally simple: it spawns subprocesses and polls artifact
   directories for exec logs. It restarts subprocesses if they exit.
"""
from __future__ import annotations

import argparse
import threading
import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ARTIFACTS_ROOT = Path('experiments') / 'artifacts'
DB_PATH = Path('experiments') / 'live_trades.db'


def ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        pair TEXT,
        timestamp TEXT,
        order_id TEXT,
        side TEXT,
        filled_size REAL,
        avg_fill_price REAL,
        fee REAL,
        notional REAL,
        strategy TEXT,
        raw_json TEXT
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        pair TEXT,
        ts TEXT,
        cash REAL,
        position REAL,
        est_value REAL
    )
    ''')

    conn.commit()
    conn.close()


def export_summary_csv(db_path: Path = DB_PATH, out_csv: Path = Path('experiments') / 'live_summary.csv') -> None:
    """Dump an aggregated summary CSV for dashboards.

    The CSV contains latest snapshot per run/pair and aggregated PnL/trade counts.
    """
    try:
        conn = sqlite3.connect(db_path)
        df_snaps = pd.read_sql_query('SELECT run_id, pair, ts, cash, position, est_value FROM snapshots', conn)
        df_trades = pd.read_sql_query('SELECT run_id, pair, COUNT(*) as n_trades, SUM(notional) as total_notional FROM trades GROUP BY run_id, pair', conn)
        if not df_snaps.empty:
            # take last snapshot per run_id/pair
            df_snaps['ts'] = pd.to_datetime(df_snaps['ts'])
            idx = df_snaps.groupby(['run_id', 'pair'])['ts'].idxmax()
            df_latest = df_snaps.loc[idx].reset_index(drop=True)
            out = df_latest.merge(df_trades, on=['run_id', 'pair'], how='left')
        else:
            out = df_trades
        out.to_csv(out_csv, index=False)
    except Exception:
        pass


def run_inprocess_trader(pair: str, ckpt: str, cash: float, poll_interval: float, conn: Optional[sqlite3.Connection] = None, initial_prices_csv: Optional[str] = None, max_notional: float = 10000.0, use_ws: bool = False, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0, take_profit_pct: float = 0.05, trailing_stop_pct: float = 0.02, max_position_size: float = 2.0, mode: str = 'replay', dry_run: bool = True, slippage_pct: float = 0.001, fee_pct: float = 0.00075):
    """Run a lightweight in-process trader that uses ModelWrapper to compute targets
    and PaperBroker to place orders. This is intentionally simple and uses a static
    price series (CSV) which it streams through.
    """
    try:
        from src.training.inference import ModelWrapper
        from src.ingestion.providers.kraken_paper import PaperBroker
    except Exception as exc:
        print(f'failed to import model or broker for inprocess trader: {exc}')
        return

    # load model
    try:
        wrapper = ModelWrapper.from_file(ckpt) if ckpt else None
    except Exception as exc:
        print(f'failed to load checkpoint {ckpt}: {exc}')
        wrapper = None

    # ensure there is a DB connection for this trader; threads must use separate connections
    local_conn = False
    if conn is None:
        try:
            ensure_db()
        except Exception:
            pass
        conn = sqlite3.connect(DB_PATH)
        local_conn = True

    # validate mode / use_ws guards
    if mode not in ('live', 'replay', 'backtest'):
        print(f'invalid mode "{mode}" for in-process trader; must be one of live|replay|backtest')
        return
    if mode == 'live' and not use_ws:
        print('mode==live requires use_ws=True to receive live ticks; aborting')
        return
    if mode != 'live' and use_ws:
        print('use_ws=True is allowed only in mode==live; aborting to avoid accidental live connections')
        return

    # optionally start Kraken WS client for live ticks
    kraken_client = None
    if use_ws:
        try:
            from src.ingestion.providers.kraken_ws import KrakenWSClient
            kraken_client = KrakenWSClient(out_root='data/raw')

            def _start_loop():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                setattr(kraken_client, '_loop', loop)
                loop.run_until_complete(kraken_client.start())
                try:
                    loop.run_forever()
                finally:
                    try:
                        loop.run_until_complete(kraken_client.stop())
                    except Exception:
                        pass

            t = threading.Thread(target=_start_loop, daemon=True)
            t.start()
            # give a moment to start
            time.sleep(0.2)
        except Exception as exc:
            print(f'failed to start KrakenWSClient, falling back to CSV/synthetic: {exc}')
            kraken_client = None

    # load prices
    if not use_ws and initial_prices_csv and Path(initial_prices_csv).exists():
        from src.utils.io import load_prices_csv
        df = load_prices_csv(initial_prices_csv, dedupe='first')
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        else:
            df.index = pd.to_datetime(df.iloc[:, 0])
        if 'close' not in df.columns:
            print('prices CSV must include close column for in-process trader')
            return
        ticks = list(zip(df.index.astype(str).to_list(), df['close'].astype(float).to_list()))
    else:
        # fallback: generate synthetic short series
        idx = pd.date_range('2020-01-01', periods=200, freq='min')
        import numpy as _np
        rng = _np.random.default_rng(1)
        returns = rng.normal(scale=0.001, size=len(idx))
        price = 100.0 + _np.cumsum(returns)
        ticks = list(zip(idx.astype(str).tolist(), price.tolist()))

    broker = PaperBroker(run_id=f'inproc-{pair}-{int(time.time())}', artifacts_root=str(ARTIFACTS_ROOT))
    position = 0.0
    cash_local = float(cash)

    price_buffer = []
    run_dir = Path(broker.artifacts_root) / broker.run_id

    i = 0
    # if using WS, we will read from kraken_client queue, otherwise iterate ticks
    while True:
        if kraken_client is not None:
            # attempt to read a message from the client's queue (thread-safe via run_coroutine_threadsafe)
            price_val = None
            ts_val = None
            try:
                q = None
                for attr in ('msg_queue', 'message_queue', 'queue'):
                    q = getattr(kraken_client, attr, None)
                    if q is not None:
                        break
                if q is not None and hasattr(kraken_client, '_loop'):
                    fut = asyncio.run_coroutine_threadsafe(q.get(), getattr(kraken_client, '_loop'))
                    try:
                        msg = fut.result(timeout=1.0)
                        if isinstance(msg, dict):
                            price_val = float(msg.get('price') or msg.get('close') or 0.0)
                            ts_val = msg.get('timestamp') or msg.get('time') or None
                    except Exception:
                        # timeout or other; skip this cycle
                        pass
            except Exception:
                pass

            if price_val is None:
                # no live message, sleep briefly
                time.sleep(poll_interval)
                continue
            ts = ts_val or datetime.utcnow().isoformat()
            price = price_val
        else:
            if i >= len(ticks):
                break
            ts, price = ticks[i]
            i += 1
        price_buffer.append({'timestamp': ts, 'close': float(price)})
        buf_df = pd.DataFrame(price_buffer)
        try:
            buf_df['timestamp'] = pd.to_datetime(buf_df['timestamp'])
            buf_df = buf_df.set_index('timestamp')
        except Exception:
            pass

        if wrapper is not None and len(buf_df) >= 3:
            try:
                targets = wrapper.predicted_to_targets(buf_df, method='vol_norm', scale=1.0, vol_window=20, max_size=2.0)
                target = float(targets.iloc[-1])
            except Exception:
                target = 0.0
        else:
            target = 0.0

        # PID-style controller + trailing-stop / take-profit
        # initialize controller state on first run
        if '_controller_state' not in globals():
            globals()['_controller_state'] = {}
        cs = globals()['_controller_state']
        if pair not in cs:
            cs[pair] = {'integrator': 0.0, 'prev_error': 0.0, 'entry_price': None, 'peak_unrealized': float('-inf'), 'last_ts': None}
        state = cs[pair]

        # desired position is the model target (clamp to allowed range)
        desired_pos = max(-max_position_size, min(max_position_size, float(target)))

        # compute dt
        now = time.time()
        if state['last_ts'] is None:
            dt = poll_interval
        else:
            dt = max(1e-6, now - state['last_ts'])
        state['last_ts'] = now

        error = desired_pos - position
        state['integrator'] += error * dt
        derivative = (error - state['prev_error']) / dt if dt > 0 else 0.0
        state['prev_error'] = error

        control = kp * error + ki * state['integrator'] + kd * derivative

        # compute trade size from control signal, convert to notional and cap by max_notional
        trade_notional = abs(control) * price
        trade_size = 0.0
        if trade_notional > 1e-12:
            scale = 1.0
            if trade_notional > max_notional:
                scale = max_notional / trade_notional
            trade_size = abs(control) * scale

        # trailing-stop / take-profit checks: compute unrealized for current position
        should_close = False
        if position != 0.0 and state.get('entry_price') is not None:
            entry = state['entry_price']
            if position > 0:
                unreal = (price - entry) / entry
            else:
                unreal = (entry - price) / entry
            # update peak
            state['peak_unrealized'] = max(state['peak_unrealized'], unreal)
            # take-profit
            if unreal >= take_profit_pct:
                should_close = True
            # trailing-stop: drop from peak beyond threshold
            if state['peak_unrealized'] - unreal >= trailing_stop_pct:
                should_close = True

        if should_close:
            # close entire position
            side = 'sell' if position > 0 else 'buy'
            size = abs(position)
            notional = size * price
            if notional > max_notional:
                # scale down close to max_notional
                scale = max_notional / notional
                size = size * scale
            if dry_run:
                order_id = f"dry-{int(time.time()*1000)}"
                filled = size
                # simulate slippage: buys pay higher price, sells receive lower price
                if side == 'buy':
                    avg_price = price * (1.0 + float(slippage_pct))
                else:
                    avg_price = price * (1.0 - float(slippage_pct))
                fee = float(fee_pct) * (filled * avg_price)
                fill = {'filled_size': filled, 'avg_fill_price': avg_price, 'fee': fee, 'order_id': order_id}
            else:
                fill = broker.place_order(pair, side, size, price=price, type='market')
                filled = float(fill.get('filled_size', 0.0))
                avg_price = float(fill.get('avg_fill_price', price))
                fee = float(fill.get('fee', 0.0) or 0.0)
            # update local cash/position
            if side == 'buy':
                cash_local -= (avg_price * filled) + fee
                position += filled
            else:
                cash_local += (avg_price * filled) - fee
                position -= filled
            # reset entry tracking
            state['entry_price'] = None
            state['peak_unrealized'] = float('-inf')
            # persist trade
            cur = conn.cursor()
            order_id = fill.get('order_id') if isinstance(fill, dict) else None
            raw = pd.Series(fill).to_json() if isinstance(fill, dict) else '{}'
            notional = filled * avg_price
            cur.execute('INSERT INTO trades (run_id, pair, timestamp, order_id, side, filled_size, avg_fill_price, fee, notional, strategy, raw_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                        (broker.run_id, pair, str(ts), order_id, side, filled, avg_price, fee, notional, 'model-inproc', raw))
            conn.commit()
        else:
            # if control requests a trade, execute partial trade
            if trade_size > 1e-12:
                side = 'buy' if control > 0 else 'sell'
                size = trade_size
                notional = size * price
                if notional > max_notional:
                    # scale down
                    scale = max_notional / notional
                    size = size * scale
                if dry_run:
                    order_id = f"dry-{int(time.time()*1000)}"
                    filled = size
                    if side == 'buy':
                        avg_price = price * (1.0 + float(slippage_pct))
                    else:
                        avg_price = price * (1.0 - float(slippage_pct))
                    fee = float(fee_pct) * (filled * avg_price)
                    fill = {'filled_size': filled, 'avg_fill_price': avg_price, 'fee': fee, 'order_id': order_id}
                else:
                    fill = broker.place_order(pair, side, size, price=price, type='market')
                    filled = float(fill.get('filled_size', 0.0))
                    avg_price = float(fill.get('avg_fill_price', price))
                    fee = float(fill.get('fee', 0.0) or 0.0)
                # update local cash/position
                if side == 'buy':
                    cash_local -= (avg_price * filled) + fee
                    position += filled
                    # set entry price if newly long
                    if state.get('entry_price') is None and filled > 0:
                        state['entry_price'] = avg_price
                        state['peak_unrealized'] = float('-inf')
                else:
                    cash_local += (avg_price * filled) - fee
                    position -= filled
                    if state.get('entry_price') is None and filled > 0:
                        state['entry_price'] = avg_price
                        state['peak_unrealized'] = float('-inf')

                # persist to DB
                cur = conn.cursor()
                order_id = fill.get('order_id') if isinstance(fill, dict) else None
                raw = pd.Series(fill).to_json() if isinstance(fill, dict) else '{}'
                notional = filled * avg_price
                cur.execute('INSERT INTO trades (run_id, pair, timestamp, order_id, side, filled_size, avg_fill_price, fee, notional, strategy, raw_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                            (broker.run_id, pair, str(ts), order_id, side, filled, avg_price, fee, notional, 'model-inproc', raw))
                conn.commit()

        # snapshot
        cur = conn.cursor()
        est_value = cash_local + position * price
        cur.execute('INSERT INTO snapshots (run_id, pair, ts, cash, position, est_value) VALUES (?,?,?,?,?,?)', (broker.run_id, pair, datetime.utcnow().isoformat(), cash_local, position, est_value))
        conn.commit()

        # export a summary CSV for dashboards
        try:
            export_summary_csv()
        except Exception:
            pass

        time.sleep(poll_interval)

    # when finished, ensure broker persisted execs
    try:
        df_exec = broker.get_exec_log()
        if not df_exec.empty:
            df_exec.to_parquet(run_dir / 'exec_log.parquet', index=False)
    except Exception:
        pass


def spawn_runner(pair: str, cash: float, max_notional: float, extra_args: Optional[List[str]] = None, *, strategy: str = 'broker', ckpt_path: Optional[str] = None) -> subprocess.Popen:
    """Spawn a runner process according to `strategy`.

    - strategy=='broker' -> spawn `tooling.run_paper_live --use-broker --live-ws`
    - strategy=='model' -> spawn `orchestration.paper_live --ckpt <ckpt> --pair <pair> --paper-mode` (model-driven)
    """
    if strategy == 'model' and ckpt_path:
        cmd = [sys.executable, '-m', 'orchestration.paper_live', '--ckpt', ckpt_path, '--pair', pair, '--no-paper-mode']
        # run in paper mode by default; `--no-paper-mode` flag in paper_live toggles live executor, so keep default
    else:
        cmd = [sys.executable, '-m', 'tooling.run_paper_live', '--use-broker', '--live-ws', '--symbol', pair, '--cash', str(cash), '--max-notional', str(max_notional)]
    if extra_args:
        cmd += extra_args
    # redirect stdout/stderr to log file per pair
    log_dir = Path('experiments') / 'live_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
    log_path = log_dir / f'{pair.replace("/", "_")}-{ts}.log'
    fh = open(log_path, 'a', buffering=1)
    p = subprocess.Popen(cmd, stdout=fh, stderr=fh)
    return p


def latest_promoted_checkpoint_for_pair(pair: str) -> Optional[str]:
    """Read `experiments/promotions.json` and return latest checkpoint path for `pair`.

    The promotions file is expected to be a JSON list of promotion records with keys
    including `pair` and `checkpoint`.
    """
    promos_path = Path('experiments') / 'promotions.json'
    if not promos_path.exists():
        return None
    try:
        with open(promos_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
    except Exception:
        return None
    # find latest entry for pair
    last = None
    for rec in data:
        if rec.get('pair') == pair and rec.get('checkpoint'):
            last = rec.get('checkpoint')
    return last


def find_new_run_dir(seen: set, timeout: int = 30) -> Optional[Path]:
    """Wait up to `timeout` seconds for a new subdirectory in ARTIFACTS_ROOT."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not ARTIFACTS_ROOT.exists():
            time.sleep(0.5)
            continue
        all_dirs = {p for p in ARTIFACTS_ROOT.iterdir() if p.is_dir()}
        new = all_dirs - seen
        if new:
            # return the most recently created
            latest = sorted(list(new), key=lambda p: p.stat().st_ctime, reverse=True)[0]
            return latest
        time.sleep(0.5)
    return None


def tail_execs_into_db(run_dir: Path, conn: sqlite3.Connection):
    """Read `exec_log.parquet` (if present) and insert any new rows into DB."""
    exec_path = run_dir / 'exec_log.parquet'
    if not exec_path.exists():
        return 0
    try:
        df = pd.read_parquet(exec_path)
    except Exception:
        # file may be being written; skip this cycle
        return 0
    inserted = 0
    cur = conn.cursor()
    run_id = run_dir.name
    # infer pair column name if present
    if 'pair' in df.columns:
        pair_col = 'pair'
    else:
        pair_col = None
    for _, row in df.iterrows():
        # dedupe by order_id (if present) and run_id
        order_id = row.get('order_id') if 'order_id' in row.index else None
        timestamp = row.get('timestamp') if 'timestamp' in row.index else None
        side = row.get('side') if 'side' in row.index else None
        filled_size = float(row.get('filled_size', 0.0) or 0.0)
        avg_fill_price = float(row.get('avg_fill_price', 0.0) or 0.0)
        fee = float(row.get('fee', 0.0) or 0.0)
        pair = row.get(pair_col) if pair_col else None
        raw = row.to_json()
        # check existing
        q = 'SELECT 1 FROM trades WHERE run_id=? AND order_id=?'
        cur.execute(q, (run_id, order_id))
        if cur.fetchone():
            continue
        cur.execute('INSERT INTO trades (run_id, pair, timestamp, order_id, side, filled_size, avg_fill_price, fee, raw_json) VALUES (?,?,?,?,?,?,?,?,?)',
                    (run_id, pair, str(timestamp), order_id, side, filled_size, avg_fill_price, fee, raw))
        inserted += 1
    conn.commit()
    return inserted


def snapshot_run_state(run_dir: Path, conn: sqlite3.Connection, initial_cash: float = 10000.0):
    # compute cash/position from trades in DB for this run
    run_id = run_dir.name
    cur = conn.cursor()
    cur.execute('SELECT pair, timestamp, side, filled_size, avg_fill_price, fee FROM trades WHERE run_id=?', (run_id,))
    rows = cur.fetchall()
    if not rows:
        return None
    pair = rows[0][0]
    cash = initial_cash
    position = 0.0
    last_price = None
    for r in rows:
        _, _, side, filled_size, avg_fill_price, fee = r
        if filled_size is None:
            continue
        if side == 'buy':
            cash -= (filled_size * avg_fill_price) + (fee or 0.0)
            position += filled_size
        else:
            cash += (filled_size * avg_fill_price) - (fee or 0.0)
            position -= filled_size
        last_price = avg_fill_price if avg_fill_price else last_price
    est_value = cash + (position * (last_price or 0.0))
    ts = datetime.utcnow().isoformat()
    cur.execute('INSERT INTO snapshots (run_id, pair, ts, cash, position, est_value) VALUES (?,?,?,?,?,?)', (run_id, pair, ts, cash, position, est_value))
    conn.commit()
    return {'run_id': run_id, 'pair': pair, 'cash': cash, 'position': position, 'est_value': est_value}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', nargs='+', required=True)
    p.add_argument('--cash', type=float, default=10000.0)
    p.add_argument('--max-notional', type=float, default=10000.0)
    p.add_argument('--strategy', choices=['broker', 'model'], default='broker', help='Strategy mode to run for each pair')
    p.add_argument('--poll-interval', type=float, default=5.0)
    p.add_argument('--mode', choices=['live', 'replay', 'backtest'], default='replay', help='Operation mode for in-process traders')
    p.add_argument('--dry-run', dest='dry_run', action='store_true')
    p.add_argument('--no-dry-run', dest='dry_run', action='store_false')
    p.set_defaults(dry_run=None)
    args = p.parse_args()

    # sensible default: if user explicitly requested live mode but didn't specify dry-run, enable dry-run
    if args.dry_run is None:
        args.dry_run = True if args.mode == 'live' else False

    ensure_db()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    # record currently existing artifact dirs
    seen = {p for p in ARTIFACTS_ROOT.iterdir() if p.is_dir()} if ARTIFACTS_ROOT.exists() else set()

    procs: Dict[str, subprocess.Popen] = {}
    run_dir_map: Dict[str, Path] = {}

    # spawn processes or in-process traders
    inproc_threads: Dict[str, threading.Thread] = {}
    for pair in args.pairs:
        ckpt = None
        if getattr(args, 'strategy', 'broker') == 'model':
            ckpt = latest_promoted_checkpoint_for_pair(pair)
            if not ckpt:
                print(f'warning: no promoted checkpoint found for {pair}; falling back to broker strategy')
        if getattr(args, 'strategy', 'broker') == 'model' and ckpt:
            # start in-process trader thread that can use WS
            # do NOT pass the shared `conn` to the thread - let each thread open its own DB connection
            t = threading.Thread(
                target=run_inprocess_trader,
                args=(pair, ckpt, args.cash, args.poll_interval),
                kwargs={
                    'initial_prices_csv': None,
                    'max_notional': args.max_notional,
                    'use_ws': True,
                    'mode': args.mode,
                    'dry_run': args.dry_run,
                },
                daemon=True,
            )
            t.start()
            inproc_threads[pair] = t
            print(f'started in-process trader for {pair} thread={t.name} (ckpt={ckpt})')
        else:
            pproc = spawn_runner(pair, args.cash, args.max_notional, strategy=getattr(args, 'strategy', 'broker'), ckpt_path=ckpt)
            procs[pair] = pproc
            print(f'spawned runner for {pair} pid={pproc.pid} strategy={getattr(args, "strategy", "broker")}')

    try:
        while True:
            # check for new artifact dirs and associate them to running procs
            for pair, proc in list(procs.items()):
                if pair in run_dir_map:
                    continue
                if proc.poll() is not None and proc.returncode is not None:
                    # process already exited before producing artifacts; attempt restart
                    print(f'process for {pair} exited early (code {proc.returncode}), restarting')
                    procs[pair] = spawn_runner(pair, args.cash, args.max_notional)
                    continue
                new = find_new_run_dir(seen, timeout=1)
                if new:
                    seen.add(new)
                    # inspect exec_log.parquet to see if it contains our pair
                    run_dir_map[pair] = new
                    print(f'paired process {pair} -> run_dir {new}')

            # tail exec logs into DB
            for pair, run_dir in list(run_dir_map.items()):
                try:
                    n = tail_execs_into_db(run_dir, conn)
                    if n:
                        print(f'inserted {n} new trades for run {run_dir.name} ({pair})')
                    # snapshot state
                    snap = snapshot_run_state(run_dir, conn, initial_cash=args.cash)
                    if snap:
                        print(f'[{pair}] cash={snap["cash"]:.2f} pos={snap["position"]:.6f} est_value={snap["est_value"]:.2f}')
                except Exception as exc:
                    print(f'error processing {run_dir}: {exc}')

            # restart any exited processes
            for pair, proc in list(procs.items()):
                if proc.poll() is not None:
                    print(f'process for {pair} died (code {proc.returncode}), restarting')
                    procs[pair] = spawn_runner(pair, args.cash, args.max_notional)

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print('stopping manager; terminating child processes')
        for p in procs.values():
            try:
                p.terminate()
            except Exception:
                pass
    finally:
        conn.close()


if __name__ == '__main__':
    main()
