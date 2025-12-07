"""Print real run artifacts for verification:
- Tail `experiments/logs/ai-crypto-controller.out.log` (last 200 lines)
- Find recent `exec_log.parquet` files under `experiments/artifacts/inproc-*` and print first 10 rows
"""
from pathlib import Path
import pandas as pd
import os
import sys

LOG_DIR = Path('experiments') / 'logs'
ARTIFACTS = Path('experiments') / 'artifacts'

def tail_file(path: Path, n: int = 200):
    if not path.exists():
        print(f'{path} not found')
        return
    try:
        with open(path, 'rb') as fh:
            fh.seek(0, os.SEEK_END)
            end = fh.tell()
            size = 1024
            data = b''
            while end > 0 and data.count(b'\n') <= n:
                start = max(0, end - size)
                fh.seek(start)
                chunk = fh.read(end - start)
                data = chunk + data
                end = start
                size *= 2
            lines = data.splitlines()[-n:]
            for l in lines:
                try:
                    print(l.decode('utf-8', errors='replace'))
                except Exception:
                    print(repr(l))
    except Exception as e:
        print('failed to tail', path, e)


def show_parquet_heads(root: Path, limit: int = 5):
    if not root.exists():
        print(f'{root} not found')
        return
    pats = sorted(list(root.glob('inproc-*')) , key=lambda p: p.stat().st_mtime, reverse=True)
    if not pats:
        print('no inproc artifact dirs found under', root)
        return
    shown = 0
    for d in pats:
        execp = d / 'exec_log.parquet'
        if not execp.exists():
            continue
        print('\n--- exec_log:', execp)
        try:
            df = pd.read_parquet(execp)
            if df.empty:
                print('empty parquet')
            else:
                print(df.head(10).to_string(index=False))
        except Exception as e:
            print('failed to read parquet', execp, e)
        shown += 1
        if shown >= limit:
            break


def main():
    # tail controller out log
    out_log = LOG_DIR / 'ai-crypto-controller.out.log'
    print('=== Tail of controller out log ===')
    tail_file(out_log, n=200)

    # show parquet heads
    print('\n=== Recent exec_log.parquet heads ===')
    show_parquet_heads(ARTIFACTS, limit=6)

if __name__ == '__main__':
    main()
