import os
import tempfile
from src.persistence.db import RunLogger, Metric, Artifact, Run
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def test_runlogger_inmemory(tmp_path):
    db_url = "sqlite:///:memory:"
    run_id = "test-run-1"
    cfg = {"x": 1}
    with RunLogger(run_id=run_id, cfg=cfg, db_url=db_url) as r:
        r.log_metrics({"sharpe": 1.23, "dd": 0.1})
        r.log_artifact("/tmp/report.json", kind="report")
        # Query counts via the same RunLogger session while it's still open
        metrics_count = r.session.query(Metric).filter(Metric.run_id == run_id).count()
        artifacts_count = r.session.query(Artifact).filter(Artifact.run_id == run_id).count()
        assert metrics_count >= 2
        assert artifacts_count >= 1
