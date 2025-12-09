"""Gap-fill CLI: find missing minute parquet files under data/raw and fill via REST OHLC.

Usage examples:

python tools/gap_fill.py --symbol XBT/USD --start 2021-01-01 --end 2021-01-03

This writes minute parquet files to `data/raw/{PAIR}/{YYYYMMDD}/{YYYYMMDDTHHMM}.parquet`.
"""
from __future__ import annotations
import argparse
import os
import logging
from datetime import datetime, timedelta
import pandas as pd

from src.ingestion.providers import kraken_rest

LOG = logging.getLogger(__name__)


def list_existing_minutes(out_root: str, pair: str, day: str):
    d = os.path.join(out_root, pair, day)
    if not os.path.exists(d):
        return set()
    s = set()
    for f in os.listdir(d):
        if not f.endswith('.parquet'):
            continue
        name = f.replace('.parquet', '')
        # expected format YYYYMMDDTHHMM
        s.add(name)
    return s


def minutes_range(start_dt: datetime, end_dt: datetime):
    cur = start_dt.replace(second=0, microsecond=0)
    while cur <= end_dt:
        yield cur
        cur = cur + timedelta(minutes=1)


def write_minute_parquet(out_root: str, pair: str, minute_dt: datetime, vwap: float, volume: float, count: int):
    out_dir = os.path.join(out_root, pair, minute_dt.strftime('%Y%m%d'))
    os.makedirs(out_dir, exist_ok=True)
    fname = minute_dt.strftime('%Y%m%dT%H%M') + '.parquet'
    path = os.path.join(out_dir, fname)
    try:
        df = pd.DataFrame([{'timestamp': minute_dt, 'vwap': float(vwap), 'volume': float(volume), 'count': int(count)}])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        tmp = path + '.tmp'
        table = pd.DataFrame(df)
        # lightweight write using pandas to_parquet if pyarrow missing
        try:
            df.to_parquet(tmp, index=False)
        except Exception:
            # fallback using pyarrow if available
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                pq.write_table(pa.Table.from_pandas(df), tmp)
            except Exception:
                LOG.exception('failed to write parquet for %s', path)
                return
        os.replace(tmp, path)
        LOG.info('wrote minute parquet %s', path)
    except Exception:
        LOG.exception('failed to write minute parquet %s', path)


def gap_fill(out_root: str, pair: str, start: datetime, end: datetime):
    # build set of existing minute identifiers
    day = start.strftime('%Y%m%d')
    cur = start
    missing = []
    for m in minutes_range(start, end):
        day = m.strftime('%Y%m%d')
        existing = list_existing_minutes(out_root, pair, day)
        key = m.strftime('%Y%m%dT%H%M')
        if key not in existing:
            missing.append(m)
    if not missing:
        LOG.info('no missing minutes between %s and %s', start, end)
        return
    LOG.info('missing %d minutes; fetching via REST', len(missing))
    # fetch a block from REST to cover the missing range (convert to epoch seconds)
    since = int(start.timestamp())
    end_ts = int(end.timestamp())
    try:
        ohlc = kraken_rest.get_ohlc(pair, '1m', since=since, end=end_ts)
    except Exception:
        LOG.exception('failed to fetch OHLC from REST for gap fill')
        return
    if ohlc is None or ohlc.empty:
        LOG.warning('REST returned no OHLC for gap fill')
        return
    ohlc = ohlc.sort_values('timestamp')
    # write each minute row as parquet
    for _, r in ohlc.iterrows():
        minute = r['timestamp'].to_pydatetime()
        write_minute_parquet(out_root, pair, minute, r.get('close', 0.0), r.get('volume', 0.0), r.get('count', 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', type=str, required=True)
    p.add_argument('--start', type=str, required=True, help='YYYY-MM-DD or ISO datetime')
    p.add_argument('--end', type=str, required=True, help='YYYY-MM-DD or ISO datetime')
    p.add_argument('--out-root', type=str, default='data/raw')
    args = p.parse_args()

    try:
        start = pd.to_datetime(args.start, utc=True).to_pydatetime()
        end = pd.to_datetime(args.end, utc=True).to_pydatetime()
    except Exception:
        LOG.exception('failed to parse dates')
        return
    gap_fill(args.out_root, args.symbol, start, end)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
