"""Run in-process traders in parallel for all available pairs under data/raw.

This script:
- lists directories under `data/raw` and converts names like `XBT_USD` -> `XBT/USD`
- for each pair finds a promoted checkpoint via `latest_promoted_checkpoint_for_pair` (fallback `models/ppo_5k.pth`)
- spawns a Thread calling `run_inprocess_trader` for each pair using the same sqlite DB connection
- waits for all threads to finish

Adjust gains/params below as needed.
"""
from threading import Thread
import sqlite3
import sys
import os
from pathlib import Path

# ensure repo root is on sys.path so we can import `scripts` as a module
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.live_paper_trader import run_inprocess_trader, latest_promoted_checkpoint_for_pair

DATA_RAW = Path('data') / 'raw'
DEFAULT_CKPT = Path('models') / 'ppo_5k.pth'

# controller params (tunable)
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
        name = p.name
        # try common encodings
        if '_' in name:
            pair = name.replace('_', '/')
        elif '-' in name:
            pair = name.replace('-', '/')
        else:
            # fallback: insert slash before last 3 chars if looks like XBTUSD
            if len(name) > 3:
                pair = name[:-3] + '/' + name[-3:]
            else:
                pair = name
        pairs.append(pair)
    return pairs


def main():
    pairs = discover_pairs()
    if not pairs:
        print('No pairs found under data/raw; aborting')
        return
    print('Found pairs:', pairs)

    threads = []
    for pair in pairs:
        ckpt = latest_promoted_checkpoint_for_pair(pair)
        if not ckpt:
            ckpt = str(DEFAULT_CKPT)
        else:
            ckpt = str(ckpt)
        print(f'starting inproc trader for {pair} using ckpt {ckpt}')
        t = Thread(
            target=run_inprocess_trader,
            args=(pair, ckpt, 10000.0, POLL),
            kwargs={
                'initial_prices_csv': 'examples/sample_prices.csv',
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

    print('All in-process traders finished')

if __name__ == '__main__':
    main()
