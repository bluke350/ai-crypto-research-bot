import argparse
import glob
import os
import pandas as pd


def find_parquet_for_pair(data_root: str, pair: str):
    pair_path = os.path.join(data_root, pair.replace('/', os.sep))
    pattern = os.path.join(pair_path, '**', 'bars.parquet')
    files = glob.glob(pattern, recursive=True)
    return files[0] if files else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pair', required=True, help='Pair e.g. XBT/USD')
    p.add_argument('--data-root', default='data/raw')
    p.add_argument('--out', default=None, help='Output CSV path')
    args = p.parse_args()

    parquet = find_parquet_for_pair(args.data_root, args.pair)
    if not parquet:
        print(f'No parquet found for {args.pair} under {args.data_root}')
        return 2

    try:
        df = pd.read_parquet(parquet)
    except Exception as e:
        print('Failed to read parquet:', e)
        return 3

    # Ensure timestamp column is ISO-8601 UTC
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        except Exception:
            pass

    out = args.out or os.path.join('examples', args.pair.replace('/', '_') + '_prices.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    try:
        df.to_csv(out, index=False)
        print('Wrote CSV:', out)
        return 0
    except Exception as e:
        print('Failed to write CSV:', e)
        return 4


if __name__ == '__main__':
    raise SystemExit(main())
