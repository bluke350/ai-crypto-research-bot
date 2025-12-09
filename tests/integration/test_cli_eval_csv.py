import sys
import subprocess
import json
import pickle
import os
from pathlib import Path


def test_cli_eval_csv(tmp_path):
    # write a tiny sample CSV
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "timestamp,close\n"
        "2020-01-01T00:00:00,100.0\n"
        "2020-01-01T00:01:00,100.1\n"
        "2020-01-01T00:02:00,100.2\n"
        "2020-01-01T00:03:00,100.15\n"
        "2020-01-01T00:04:00,100.3\n"
    )

    # create a minimal checkpoint the CLI can load
    ckpt_path = tmp_path / "ckpt.pkl"
    ck = {"model": {"coef": [0.1, -0.05]}, "type": "linear_baseline"}
    with open(ckpt_path, "wb") as fh:
        pickle.dump(ck, fh)

    outdir = tmp_path / "out"
    outdir.mkdir()

    cmd = [
        sys.executable,
        "orchestration/cli.py",
        "eval",
        "--ckpt",
        str(ckpt_path),
        "--prices-csv",
        str(csv_path),
        "--timestamp-col",
        "timestamp",
        "--out",
        str(outdir),
    ]

    env = os.environ.copy()
    # ensure repo root is on PYTHONPATH so orchestration/cli.py can import `src`
    env["PYTHONPATH"] = str(Path.cwd())
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise AssertionError(f"CLI failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    summary = outdir / "eval_summary.json"
    assert summary.exists(), "eval_summary.json not written"
    data = json.loads(summary.read_text())
    assert "pnl_head" in data and "n_executions" in data
