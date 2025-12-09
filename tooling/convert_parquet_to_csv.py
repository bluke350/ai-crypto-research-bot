import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument('parquet')
    p.add_argument('csv')
    args = p.parse_args()
    df = pd.read_parquet(args.parquet)
    # Ensure expected columns: timestamp, open, high, low, close, volume
    if 'timestamp' not in df.columns:
        if 'time' in df.columns:
            df = df.rename(columns={'time':'timestamp'})
        else:
            # try to use index
            try:
                df = df.reset_index()
            except Exception:
                pass
    cols = ['timestamp','open','high','low','close','volume']
    present = [c for c in cols if c in df.columns]
    df = df[present]
    df.to_csv(args.csv, index=False)
    print(f"Wrote {len(df)} rows to {args.csv}")

if __name__ == '__main__':
    main()
