import os
import tempfile
from datetime import datetime, timedelta
import traceback

from src.ingestion.providers.kraken_ws import KrakenWSClient


def run():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["WS_WAL_COMPRESS_DAYS"] = "0"
        os.environ["WS_WAL_COMPRESS_INTERVAL_HOURS"] = "0.001"
        os.environ["WS_WAL_RETENTION_DAYS"] = "0"
        os.environ["WS_WAL_PRUNE_INTERVAL_HOURS"] = "0.001"

        out = os.path.join(tmp, "data")
        client = KrakenWSClient(out_root=out)
        archive_root = os.path.join(out, "_wal", "archive", "XBT/USD")
        old_day = (datetime.utcnow() - timedelta(days=2)).strftime("%Y%m%d")
        old_dir = os.path.join(archive_root, old_day)
        os.makedirs(old_dir, exist_ok=True)
        # create a fake parquet file inside
        with open(os.path.join(old_dir, "f.parquet"), "w") as f:
            f.write("data")

        try:
            # Call compress and prune helpers
            client.compress_archived_wal_once()
            client.prune_archived_wal_once()

            tar = old_dir + ".tar.gz"
            print("old_dir_exists", os.path.exists(old_dir))
            print("tar_exists", os.path.exists(tar))
            if (not os.path.exists(old_dir)) or (not os.path.exists(tar)):
                print("PASS")
            else:
                print("FAIL: both dir and tar.gz still exist")
        except Exception:
            traceback.print_exc()


if __name__ == '__main__':
    run()
