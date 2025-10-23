import tempfile
import os
from src.ingestion.providers.kraken_paper import PaperBroker


def test_paper_broker_logs(tmp_path):
    run_id = "r1"
    artifacts = tmp_path / "artifacts"
    b = PaperBroker(run_id=run_id, artifacts_root=str(artifacts))
    fill = b.place_order("XBT/USD", "buy", 0.01, price=50000.0)
    assert fill["status"] == "filled"
    path = artifacts / run_id / "exec_log.parquet"
    assert path.exists()
