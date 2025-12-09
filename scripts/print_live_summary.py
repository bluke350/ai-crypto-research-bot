import sqlite3
import pandas as pd
import os
from pathlib import Path

DB = Path('experiments') / 'live_trades.db'
CSV = Path('experiments') / 'live_summary.csv'

print('DB exists:', DB.exists())
if not DB.exists():
    raise SystemExit('DB not found')

conn = sqlite3.connect(DB)
try:
    df_tr = pd.read_sql_query('SELECT run_id, pair, COUNT(*) as n_trades, SUM(notional) as total_notional FROM trades GROUP BY run_id, pair', conn)
    print('\nTrades summary:')
    print(df_tr.to_string(index=False))
except Exception as e:
    print('reading trades failed:', e)

try:
    df_sn = pd.read_sql_query('SELECT run_id, pair, ts, cash, position, est_value FROM snapshots', conn)
    if not df_sn.empty:
        df_sn['ts'] = pd.to_datetime(df_sn['ts'])
        idx = df_sn.groupby(['run_id', 'pair'])['ts'].idxmax()
        print('\nLatest snapshots (last per run/pair):')
        print(df_sn.loc[idx].reset_index(drop=True).to_string(index=False))
    else:
        print('\nNo snapshots found')
except Exception as e:
    print('reading snapshots failed:', e)

print('\nCSV exists:', CSV.exists())
if CSV.exists():
    print('\nCSV head:')
    with open(CSV, 'r', encoding='utf-8') as fh:
        for i, line in enumerate(fh):
            if i >= 20:
                break
            print(line.rstrip('\n'))

conn.close()
