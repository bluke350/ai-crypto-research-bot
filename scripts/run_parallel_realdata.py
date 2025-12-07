"""
Run in-process traders in parallel using real parquet files under `data/raw`.
For each discovered pair, find the latest parquet file and convert it to a temp CSV
under `experiments/manual_prices_from_parquet/{pair.replace('/','_')}.csv`, then
start `run_inprocess_trader` reading that CSV.

This proves traders ran on real historic prices (not synthetic 100.0 series).
"""
from threading import Thread
import os
import sys
from pathlib import Path
import sqlite3
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.live_paper_trader import run_inprocess_trader, latest_promoted_checkpoint_for_pair

DATA_RAW = Path('data') / 'raw'
OUT_DIR = Path('experiments') / 'manual_prices_from_parquet'
DEFAULT_CKPT = Path('models') / 'ppo_5k.pth'

# controller params
KP = 0.6
KI = 0.15
KD = 0.01
TP = 0.03
TS = 0.01
MAX_NOTIONAL = 500.0
MAX_POS = 1.0
POLL = 1.0


def discover_pairs():
    if not DATA_RAW.exists():
        return []
    pairs = []
    for p in sorted(DATA_RAW.iterdir()):
        if not p.is_dir():
            continue
        # try to normalize name like earlier code
        name = p.name
        if '_' in name:
            pair = name.replace('_', '/')
        elif '-' in name:
            pair = name.replace('-', '/')
        else:
            if len(name) > 3:
                pair = name[:-3] + '/' + name[-3:]
            else:
                pair = name
        pairs.append((pair, p))
    return pairs


def find_latest_parquet(pair_dir: Path) -> Path | None:
    # find newest .parquet under pair_dir recursively
    files = list(pair_dir.rglob('*.parquet'))
    if not files:
        return None
    latest = max(files, key=lambda p: p.stat().st_mtime)
    return latest


def prepare_csv_from_parquet(parquet_path: Path, out_csv: Path, max_rows: int = None) -> bool:
    try:
        df = pd.read_parquet(parquet_path)
        # normalize timestamp and required columns
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'])
        elif 'ts' in df.columns:
            df['timestamp'] = pd.to_datetime(df['ts'])
        else:
            # try index
            if df.index.dtype.kind in 'M':
                df = df.reset_index().rename(columns={'index': 'timestamp'})
            else:
                # give up
                return False
        if 'close' not in df.columns and 'price' in df.columns:
            df['close'] = df['price']
        if 'close' not in df.columns:
            return False
        # take last N rows if requested
        if max_rows is not None:
            df = df.tail(max_rows)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df[['timestamp','open','high','low','close','volume']].to_csv(out_csv, index=False)
        return True
    except Exception as e:
        print('failed to convert parquet to csv', parquet_path, e)
        return False


def main():
    pairs = discover_pairs()
    if not pairs:
        print('no pairs under data/raw')
        return
    print('discovered pairs:', [p for p,_ in pairs])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    threads = []
    for pair, dirpath in pairs:
        pq = find_latest_parquet(dirpath)
        if pq is None:
            print('no parquet for', pair, 'skipping')
            continue
        out_csv = OUT_DIR / f"{pair.replace('/','_')}.csv"
        ok = prepare_csv_from_parquet(pq, out_csv, max_rows=2000)
        if not ok:
            print('failed to prepare csv for', pair, 'from', pq)
            continue
        ckpt = latest_promoted_checkpoint_for_pair(pair)
        if not ckpt:
            ckpt = str(DEFAULT_CKPT)
        else:
            ckpt = str(ckpt)
        print(f'starting inproc trader for {pair} using ckpt {ckpt} and prices {out_csv}')
        t = Thread(
            target=run_inprocess_trader,
            args=(pair, ckpt, 10000.0, POLL),
            kwargs={
                'initial_prices_csv': str(out_csv),
                'max_notional': MAX_NOTIONAL,
                'kp': KP,
                'ki': KI,
                'kd': KD,
                'take_profit_pct': TP,
                'trailing_stop_pct': TS,
                'max_position_size': MAX_POS,
                'use_ws': False,
                'mode': 'replay',
                'dry_run': True,
            },
            daemon=False,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    print('finished all inproc traders')

if __name__ == '__main__':
    main()
