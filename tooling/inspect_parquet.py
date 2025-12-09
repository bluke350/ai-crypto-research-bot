import glob
import os
import pandas as pd


def inspect_parquets(root="data/raw"):
    files = glob.glob(os.path.join(root, "**", "bars.parquet"), recursive=True)
    if not files:
        print("No bars.parquet files found under", root)
        return 1

    for f in sorted(files):
        print("==", f)
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            print("ERROR reading:", repr(e))
            continue

        print("rows:", len(df))
        print("cols:", list(df.columns))
        if "timestamp" in df.columns:
            try:
                tsmin = df["timestamp"].min()
                tsmax = df["timestamp"].max()
                print("timestamp range:", tsmin, "->", tsmax)
            except Exception:
                pass

        # Show first 5 rows compactly
        with pd.option_context('display.max_columns', 20, 'display.width', 200):
            print(df.head().to_string(index=False))
        print()

    return 0


if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    sys.exit(inspect_parquets(root))
