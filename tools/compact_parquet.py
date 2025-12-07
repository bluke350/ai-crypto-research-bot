import os
import logging
import pyarrow.parquet as pq
import pandas as pd

logger = logging.getLogger(__name__)


def compact_day(dir_path: str):
    # read all parquet files in dir_path and write daily.parquet
    files = [f for f in os.listdir(dir_path) if f.endswith('.parquet') and not f.endswith('.tmp') and f != 'daily.parquet']
    dfs = []
    for f in files:
        p = os.path.join(dir_path, f)
        try:
            df = pq.read_table(p).to_pandas()
            dfs.append(df)
        except Exception:
            logger.exception('failed to read parquet %s', p)
            continue
    if not dfs:
        return None
    all_df = pd.concat(dfs).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    out = os.path.join(dir_path, 'daily.parquet')
    try:
        pq.write_table(pq.Table.from_pandas(all_df), out)
    except Exception:
        logger.exception('failed to write daily parquet %s', out)
        raise
    return out


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: compact_parquet.py <day_dir>')
        raise SystemExit(1)
    print('Writing', compact_day(sys.argv[1]))
