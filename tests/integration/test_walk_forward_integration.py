import json
import os
import sqlite3
import subprocess
from pathlib import Path


def test_walk_forward_smoke_with_rl_checkpoint(tmp_path):
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir()
    db_path = tmp_path / "registry.db"

    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + os.pathsep + "."
    env["EXPERIMENT_DB_URL"] = f"sqlite:///{db_path}"

    cmd = [
        "python",
        "orchestration/pipelines/walk_forward_pipeline.py",
        "--prices-csv",
        "examples/XBT_USD_prices.csv",
        "--seeds",
        "0",
        "--ppo-checkpoint",
        "models/ppo_smoke.pth",
        "--output",
        str(out_dir),
        "--register",
    ]

    res = subprocess.run(cmd, cwd=os.getcwd(), env=env, capture_output=True, text=True)
    print(res.stdout)
    print(res.stderr)
    assert res.returncode == 0, f"pipeline failed: {res.stderr}"

    # find results.json under out_dir
    runs = list(out_dir.iterdir())
    assert runs, "no run outputs created"
    run_dir = runs[0]
    results_file = run_dir / "results.json"
    assert results_file.exists(), "results.json not written"

    j = json.loads(results_file.read_text())
    # expect seeds key with non-empty folds
    assert "seeds" in j and len(j["seeds"]) > 0
    assert j["seeds"][0]["n_folds"] > 0

    # verify registration in sqlite DB
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM artifacts WHERE kind = ?", ("walk_forward",))
    count = cur.fetchone()[0]
    conn.close()
    assert count >= 1, "no artifact registration found in registry DB"
