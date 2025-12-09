from pathlib import Path
import pandas as pd
import sys

DATA_RAW = Path('data') / 'raw'
if not DATA_RAW.exists():
    print('data/raw not found')
    sys.exit(1)

pairs = []
for p in sorted(DATA_RAW.iterdir()):
    if not p.is_dir():
        continue
    name = p.name
    if '_' in name:
        pair = name.replace('_','/')
    elif '-' in name:
        pair = name.replace('-','/')
    else:
        if len(name) > 3:
            pair = name[:-3] + '/' + name[-3:]
        else:
            pair = name
    pairs.append((pair,p))

if not pairs:
    print('no pairs')
    sys.exit(0)

for pair, d in pairs:
    # find newest parquet under d
    files = list(d.rglob('*.parquet'))
    if not files:
        print(pair, 'NO PARQUET')
        continue
    latest = max(files, key=lambda p: p.stat().st_mtime)
    print('\nPAIR:', pair, 'latest parquet:', latest)
    try:
        df = pd.read_parquet(latest)
    except Exception as e:
        print('  failed to read parquet:', e)
        continue
    # find close-like column
    col = None
    for c in ('close','price','close_price'):
        if c in df.columns:
            col = c
            break
    if col is None:
        # maybe it's in a nested structure or index
        print('  no close column; columns:', list(df.columns)[:10])
        continue
    # print last 5 rows
    last = df[[col]].tail(5)
    try:
        last.index = pd.to_datetime(last.index)
    except Exception:
        pass
    print(last)
