import json
import os
import shutil
import subprocess
import sys
import tempfile


def test_controller_once_creates_artifacts(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    # run controller in once mode against sample csv
    cmd = [
        sys.executable,
        "-m",
        "scripts.continuous_controller",
        "--once",
        "--prices-csv",
        "examples/sample_prices_for_cli.csv",
        "--n-trials",
        "1",
        "--artifacts-root",
        str(artifacts),
    ]
    res = subprocess.run(cmd, check=True)
    # assert that a results.json file exists somewhere under artifacts
    found = False
    for root, dirs, files in os.walk(artifacts):
        if "results.json" in files:
            found = True
            break
    assert found, "tuning did not produce results.json under artifacts"
