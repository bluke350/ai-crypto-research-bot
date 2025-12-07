from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.execution.order_executor import PaperOrderExecutor, LiveAdapterOrderExecutor
from src.execution.order_models import Order
from src.execution.simulators import ExchangeSimulator
from src.training.inference import ModelWrapper
from src.ingestion.providers.kraken_ws import KrakenWSClient
from src.execution.gating import RiskGate, RiskConfig
import threading
import asyncio
from tooling.structured_logging import setup_structured_logger
from orchestration.ws_parallel import AsyncWorkerPool

# structured logger per-run will be created when out_dir is known; use module logger
logger = setup_structured_logger(__name__)

# Prometheus metrics (optional)
try:
    from prometheus_client import start_http_server, Counter, Gauge
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False


def run_live(checkpoint_path: str,
             prices_csv: Optional[str] = None,
             out_root: str = "experiments/artifacts",
             pair: str = "XBT/USD",
             sim_seed: Optional[int] = None,
             max_ticks: Optional[int] = None,
             stream_delay: float = 0.0,
             sleep_between: bool = False,
             sizing_mode: str = "units",
             use_ws: bool = False,
             paper_mode: bool = True,
             max_position: float = 10.0,
             max_order_notional: float = 10000.0,
             max_loss: float = 10000.0,
             target_smoothing_alpha: float = 0.3,
             max_delta_per_tick: float = 1.0,
             prom_port: int | None = None,) -> str:
    """Run a paper-mode live streamer using either a CSV file (stream) or synthetic ticks.

    This runner processes ticks one-by-one, uses the model to produce a target at each step,
    computes the delta vs current position and sends an order to the PaperOrderExecutor.

    Parameters:
    - checkpoint_path: path to model checkpoint (pickle)
    - prices_csv: optional CSV path with timestamp and close columns
    - out_root: directory to write artifacts
    - pair, sim_seed: simulator settings
    - max_ticks: stop after this many ticks (useful for testing)
    - stream_delay: seconds to sleep between ticks (0 for no sleep)
    - sleep_between: if True use time.sleep(stream_delay) between ticks

    Returns path to run artifacts directory.
    """
    run_id = str(uuid.uuid4())
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # write run plan / reproducibility snapshot
    run_plan = {
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat(),
        "checkpoint_path": checkpoint_path,
        "pair": pair,
        "sim_seed": sim_seed,
        "use_ws": bool(use_ws),
        "max_ticks": max_ticks,
        "stream_delay": stream_delay,
        "sizing_mode": sizing_mode,
        "safety": {
            "max_position": max_position,
            "max_order_notional": max_order_notional,
            "max_loss": max_loss,
        },
        "env": {k: os.environ.get(k) for k in [
            "ENABLE_LIVE_EXECUTOR", "WS_DISABLE_HEALTH_SERVER", "WS_BACKOFF_POLICY",
        ]}
    }
    try:
        with open(out_dir / 'run_plan.json', 'w', encoding='utf-8') as fh:
            json.dump(run_plan, fh, indent=2)
    except Exception:
        logger.exception('failed to write run_plan.json')

    # reconfigure logger to write to run-specific file (structured logger supports file handler)
    try:
        log_path = out_dir / 'run.log'
        # create a run-scoped logger that writes to the run log file
        run_logger = setup_structured_logger(f"{__name__}.{run_id}", file_path=str(log_path))
    except Exception:
        # fallback to module logger
        run_logger = logger
        logger.exception('failed to configure run-specific logger')

    # load model
    wrapper = ModelWrapper.from_file(checkpoint_path)

    # data source: read CSV fully but we'll stream rows; if not provided, synthetic
    if prices_csv:
        from src.utils.io import load_prices_csv
        df = load_prices_csv(prices_csv, dedupe='first')
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        else:
            # assume first column is timestamp
            first = df.columns[0]
            df[first] = pd.to_datetime(df[first])
            df = df.set_index(first)
        if 'close' not in df.columns:
            raise ValueError('prices CSV must include close column')
        ticks = list(zip(df.index.astype(str).to_list(), df['close'].astype(float).to_list()))
    else:
        # generate synthetic series
        idx = pd.date_range('2020-01-01', periods=200, freq='T')
        import numpy as _np

        rng = _np.random.default_rng(0)
        returns = rng.normal(scale=0.001, size=len(idx))
        price = 100.0 + _np.cumsum(returns)
        ticks = list(zip(idx.astype(str).tolist(), price.tolist()))

    sim = ExchangeSimulator(pair=pair, seed=sim_seed)
    # start prometheus metrics server if requested and available
    metrics = {}
    if prom_port and PROM_AVAILABLE:
        try:
            start_http_server(prom_port)
            metrics['orders_executed'] = Counter('orders_executed', 'Number of orders executed')
            metrics['orders_rejected'] = Counter('orders_rejected', 'Number of orders rejected by safety checks')
            metrics['circuit_breaker'] = Counter('circuit_breaker_activations', 'Number of times circuit breaker activated')
            metrics['current_position'] = Gauge('current_position', 'Current position')
            metrics['portfolio_value'] = Gauge('portfolio_value', 'Current portfolio value')
            metrics['realized_pnl'] = Gauge('realized_pnl', 'Realized PnL')
            metrics['avg_entry_price'] = Gauge('avg_entry_price', 'Average entry price')
        except Exception:
            logger.exception('failed to start prometheus server')

    # choose executor: by default run in paper_mode (PaperOrderExecutor) even when using live WS
    if not paper_mode:
        enable_live = os.environ.get('ENABLE_LIVE_EXECUTOR', '0') == '1'
    else:
        enable_live = False

    if enable_live:
        # LiveAdapterOrderExecutor will default to dry_run unless a client is provided
        executor = LiveAdapterOrderExecutor(client=None, dry_run=True)
    else:
        executor = PaperOrderExecutor(simulator=sim)

    cash = 1_000_000.0
    position = 0.0
    avg_entry_price = 0.0
    realized_pnl = 0.0
    pnl_history = []
    exec_rows = []
    smoothed_targets = []

    # instantiate risk gate from run parameters
    gate_cfg = RiskConfig(max_position_abs=float(max_position), max_order_notional=float(max_order_notional), daily_loss_limit=float(max_loss))
    gate = RiskGate(gate_cfg)

    # buffer of recent prices for model features
    price_buffer = []

    # If WS requested, start KrakenWSClient in background thread
    kraken_client = None
    if use_ws:
        try:
            kraken_client = KrakenWSClient(out_root='data/raw')

            def _start_loop():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                # expose loop on client so main thread can post coroutines
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
            # give it a moment to start
            time.sleep(0.1)
            run_logger.info('kraken ws client started in background thread')

            # create an async worker pool to process messages concurrently (bounded parallelism)
            try:
                loop = getattr(kraken_client, '_loop', None)
                if loop is not None:
                    # handler coroutine: transform msg and push into price_buffer in main thread via thread-safe call
                    async def _handle_msg(msg):
                        # attempt to extract price and timestamp
                        try:
                            p = float(msg.get('price') or msg.get('close'))
                        except Exception:
                            p = None
                        ts = msg.get('timestamp') or msg.get('time') or None
                        # push a tuple directly into the client's queue for main loop consumption
                        # We will put into kraken_client.msg_queue already in _connect_loop; here we just process/transform
                        # For compatibility, attach a normalized dict back onto the msg for consumers
                        if p is not None:
                            msg['price'] = p
                        if ts is not None:
                            msg['timestamp'] = ts

                    # instantiate worker pool on client's loop
                    def _start_pool():
                        # run the worker pool coroutine in the kraken loop
                        pool = AsyncWorkerPool(kraken_client.msg_queue, _handle_msg, num_workers=4)
                        setattr(kraken_client, '_worker_pool', pool)
                        # schedule pool.start() on the client's loop without blocking
                        try:
                            import asyncio as _asyncio

                            _asyncio.run_coroutine_threadsafe(pool.start(), loop)
                        except Exception:
                            # best-effort: if scheduling fails, ignore and continue
                            pass

                    t2 = threading.Thread(target=_start_pool, daemon=True)
                    t2.start()
            except Exception:
                run_logger.exception('failed to start async worker pool for ws messages')
        except Exception:
            run_logger.exception('failed to start kraken ws client; falling back to CSV/synthetic')

    for i, (ts, price) in enumerate(ticks):
        # If using WS, attempt to fetch a message from the kraken client's queue
        if use_ws and kraken_client is not None:
            try:
                q = None
                for attr in ('msg_queue', 'message_queue', 'queue'):
                    q = getattr(kraken_client, attr, None)
                    if q is not None:
                        break
                if q is not None and hasattr(kraken_client, '_loop'):
                    fut = asyncio.run_coroutine_threadsafe(q.get(), getattr(kraken_client, '_loop'))
                    try:
                        msg = fut.result(timeout=0.5)
                        # msg may contain price info; attempt to extract
                        if isinstance(msg, dict):
                            price = msg.get('price') or msg.get('close') or price
                            ts = msg.get('timestamp') or ts
                        else:
                            # leave price as-is
                            pass
                    except Exception:
                        # timeout or other, fall back to CSV/synthetic tick
                        pass
            except Exception:
                run_logger.debug('ws message fetch failed; continuing with CSV/synthetic tick')
        # append to buffer
        price_buffer.append({'timestamp': ts, 'close': float(price)})
        buf_df = pd.DataFrame(price_buffer)
        # ensure index is datetime for model utils
        try:
            buf_df['timestamp'] = pd.to_datetime(buf_df['timestamp'])
            buf_df = buf_df.set_index('timestamp')
        except Exception:
            pass

        # require at least 3 rows for the ModelWrapper prediction logic
        if len(buf_df) >= 3:
            # predict target for the most recent timestamp using ModelWrapper.predicted_to_targets
            targets = wrapper.predicted_to_targets(buf_df, method='vol_norm', scale=1.0, vol_window=20, max_size=2.0)
            # targets index aligned to buf_df.index[2:]; get last target
            try:
                target = float(targets.iloc[-1])
            except Exception:
                target = 0.0
        else:
            target = 0.0

        # apply simple exponential smoothing to avoid churn
        if smoothed_targets:
            prev = float(smoothed_targets[-1][1])
            smoothed = float(prev * float(target_smoothing_alpha) + float(target) * (1.0 - float(target_smoothing_alpha)))
        else:
            smoothed = float(target)
        smoothed_targets.append((ts, smoothed))

        # compute delta and send order if needed (cap per-tick change)
        delta = smoothed - position
        # cap delta magnitude to max_delta_per_tick
        if abs(delta) > float(max_delta_per_tick):
            capped = float(max_delta_per_tick) if delta > 0 else -float(max_delta_per_tick)
            run_logger.info('delta capped from %.6f to %.6f (max_delta_per_tick=%.6f)', delta, capped, float(max_delta_per_tick))
            delta = capped
        if abs(delta) > 1e-12:
            side = 'buy' if delta > 0 else 'sell'
            order = Order(order_id=f'o{i}', pair=pair, side=side, size=delta, price=None)
            # risk gate checks (position, per-order notional, daily loss)
            positions = {pair: float(position)}
            allowed, reason = gate.check_order(order, market_price=price, positions=positions, realized_pnl=0.0)
            if not allowed:
                run_logger.warning('order rejected by gate: %s', reason, extra={"order_id": order.order_id})
                if metrics.get('orders_rejected') is not None:
                    metrics['orders_rejected'].inc()
                # record a rejected execution row for auditing
                fill = {"filled_size": 0.0, "avg_fill_price": float(price), "fee": 0.0, "gate_allowed": False, "gate_reason": reason}
                # if gate tripped due to daily loss, activate circuit breaker
                if gate.tripped:
                    run_logger.error('risk gate tripped: %s', gate.trip_reason)
                    if metrics.get('circuit_breaker') is not None:
                        metrics['circuit_breaker'].inc()
                    break
            else:
                # allowed by gate -> execute
                fill = executor.execute(order, market_price=price, is_maker=False)
                # annotate executed fill with gate info
                try:
                    fill['gate_allowed'] = True
                    fill['gate_reason'] = None
                except Exception:
                    pass
            filled = float(fill.get('filled_size', 0.0))
            avg_price = float(fill.get('avg_fill_price', price))
            fee = float(fill.get('fee', 0.0))
            # compute realized PnL when trades close existing positions
            try:
                pos_prev = float(position)
                exec_size = float(filled)
                new_pos = pos_prev + exec_size
                # closing amount is when exec and pos have opposite signs
                if pos_prev != 0 and (pos_prev * exec_size) < 0:
                    closed = min(abs(exec_size), abs(pos_prev))
                    if pos_prev > 0:
                        # closing long: profit = closed * (exec_price - avg_entry_price)
                        realized_pnl += closed * (avg_price - avg_entry_price)
                    else:
                        # closing short: profit = closed * (avg_entry_price - exec_price)
                        realized_pnl += closed * (avg_entry_price - avg_price)
                # update average entry price for resulting open position
                if new_pos == 0:
                    avg_entry_price = 0.0
                else:
                    # if trade increases/creates position on same side, update weighted avg
                    if pos_prev == 0 or (pos_prev * exec_size) > 0:
                        total_qty = abs(pos_prev) + abs(exec_size)
                        if total_qty > 0:
                            avg_entry_price = ((abs(pos_prev) * avg_entry_price) + (abs(exec_size) * avg_price)) / total_qty
                    else:
                        # trade reduced or flipped; if flipped, set entry price to exec price for remaining
                        if abs(new_pos) > 0 and (pos_prev * new_pos) < 0:
                            # flipped: remaining (abs(new_pos)) is from this execution at exec price
                            avg_entry_price = avg_price
                # update cash/position
                cash -= (avg_price * filled) + fee
                position = new_pos
            except Exception:
                # fallback: simple update
                cash -= (avg_price * filled) + fee
                position += filled
            row = {**dict(fill), 'timestamp': ts, 'requested_size': delta, 'side': side}
            exec_rows.append(row)
            # update metrics and safety checks
            if metrics.get('orders_executed') is not None:
                metrics['orders_executed'].inc()
                try:
                    metrics['current_position'].set(float(position))
                except Exception:
                    pass
            # enforce max position circuit breaker
            if abs(position) > float(max_position):
                run_logger.error('circuit breaker: position %.4f exceeds max_position %.4f', position, float(max_position))
                if metrics.get('circuit_breaker') is not None:
                    metrics['circuit_breaker'].inc()
                break

        # mark to market
        value = cash + position * price
        pnl_history.append((ts, float(value)))
        if metrics.get('portfolio_value') is not None:
            try:
                metrics['portfolio_value'].set(float(value))
            except Exception:
                pass
        # update realized PnL / avg entry price gauges if available
        if metrics.get('realized_pnl') is not None:
            try:
                metrics['realized_pnl'].set(float(realized_pnl))
            except Exception:
                pass
        if metrics.get('avg_entry_price') is not None:
            try:
                metrics['avg_entry_price'].set(float(avg_entry_price))
            except Exception:
                pass

        # global drawdown circuit breaker (simple check vs starting capital)
        if float(1_000_000.0 - value) > float(max_loss):
            run_logger.error('global drawdown exceeded: loss %.2f > max_loss %.2f — activating circuit breaker', (1_000_000.0 - value), float(max_loss))
            if metrics.get('circuit_breaker') is not None:
                metrics['circuit_breaker'].inc()
            break

        # optionally sleep to simulate real-time
        if sleep_between and stream_delay > 0:
            time.sleep(stream_delay)

        if max_ticks is not None and (i + 1) >= int(max_ticks):
            break

    # write results
    serial = {'pnl': pnl_history, 'executions': exec_rows}
    # include smoothed targets for debugging/analysis
    try:
        serial['smoothed_targets'] = [(t, float(v)) for t, v in smoothed_targets]
    except Exception:
        serial['smoothed_targets'] = []
    # include executor fill records for audit if available
    executions = serial.get('executions', [])
    if hasattr(executor, 'fills'):
        try:
            executions = executions + list(getattr(executor, 'fills'))
        except Exception:
            run_logger.exception('failed to read executor.fills')
    serial['executions'] = executions

    with open(out_dir / 'result.json', 'w', encoding='utf-8') as fh:
        json.dump(serial, fh, indent=2)

    summary = {'run_id': run_id, 'created_at': datetime.utcnow().isoformat(), 'n_executions': len(serial.get('executions', []))}
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)

    # persist gate state for auditing
    try:
        gate_state = {
            'tripped': getattr(gate, 'tripped', False),
            'trip_reason': getattr(gate, 'trip_reason', None),
            'realized_pnl': float(realized_pnl),
            'avg_entry_price': float(avg_entry_price),
        }
        with open(out_dir / 'gate_state.json', 'w', encoding='utf-8') as fh:
            json.dump(gate_state, fh, indent=2)
    except Exception:
        run_logger.exception('failed to write gate_state.json')

    return str(out_dir)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--prices-csv', default=None)
    p.add_argument('--out-root', default='experiments/artifacts')
    p.add_argument('--pair', default='XBT/USD')
    p.add_argument('--sim-seed', type=int, default=None)
    p.add_argument('--max-ticks', type=int, default=None)
    p.add_argument('--stream-delay', type=float, default=0.0)
    p.add_argument('--sleep-between', action='store_true', help='sleep between ticks to simulate real-time')
    p.add_argument('--use-ws', action='store_true', help='use Kraken WS feed if available')
    p.add_argument('--no-paper-mode', dest='paper_mode', action='store_false', help='disable paper-mode and allow live executor when enabled')
    p.add_argument('--live-executor', dest='live_executor', action='store_true', help='(danger) enable live executor adapter (requires env flags & credentials)')
    p.add_argument('--prom-port', type=int, default=None, help='start prometheus metrics server on this port')
    p.add_argument('--max-position', type=float, default=10.0, help='max absolute position before circuit breaker')
    p.add_argument('--max-order-notional', type=float, default=10000.0, help='max notional per order')
    p.add_argument('--max-loss', type=float, default=10000.0, help='max unrealized loss before circuit breaker')
    args = p.parse_args()
    print('running paper live runner...')
    print(run_live(
        args.ckpt,
        prices_csv=args.prices_csv,
        out_root=args.out_root,
        pair=args.pair,
        sim_seed=args.sim_seed,
        max_ticks=args.max_ticks,
        stream_delay=args.stream_delay,
        sleep_between=args.sleep_between,
        use_ws=args.use_ws,
        paper_mode=args.paper_mode,
        prom_port=args.prom_port,
        max_position=args.max_position,
        max_order_notional=args.max_order_notional,
        max_loss=args.max_loss,
    ))
