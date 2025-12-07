import sys
import subprocess
import json
import os
import pickle
from pathlib import Path


def test_auto_run_smoke(tmp_path):
    csv_path = tmp_path / "sample.csv"
    # generate a short synthetic CSV with 60 rows
    lines = ["timestamp,close"]
    from datetime import datetime, timedelta
    t0 = datetime(2020, 1, 1)
    price = 100.0
    for i in range(60):
        ts = (t0 + timedelta(minutes=i)).isoformat()
        price = price + (0.01 if i % 2 == 0 else -0.005)
        lines.append(f"{ts},{price}")
    csv_path.write_text("\n".join(lines))

    outdir = tmp_path / "out"
    outdir.mkdir()

    cmd = [
        sys.executable,
        "-m",
        "orchestration.auto_run",
        "--prices-csv",
        str(csv_path),
        "--out-root",
        str(outdir),
        "--steps",
        "5",
        "--seed",
        "0",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    env["WS_DISABLE_HEALTH_SERVER"] = "1"

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise AssertionError(f"auto_run failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    # find run folder
    runs = list(outdir.iterdir())
    assert runs, "no run directories created"
    run_dir = runs[0]
    summary = run_dir / "summary.json"
    assert summary.exists(), "summary.json not created"
    data = json.loads(summary.read_text())
    assert "run_id" in data
    assert "max_drawdown" in data
    assert "n_executions" in data
