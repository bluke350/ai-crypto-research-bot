import os
import sqlite3
import pandas as pd
import sys
import pytest

from orchestration.pipelines import training_pipeline as tp


def test_training_pipeline_register_db(tmp_path, monkeypatch):
    # prepare minimal run settings: run trainer with a tiny step count
    db_path = tmp_path / 'registry.db'
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv('EXPERIMENT_DB_URL', db_url)
    argv = ["training_pipeline.py", "--model", "rl", "--steps", "10", "--save", str(tmp_path / 'models' / 'ppo.pth'), "--seeds", "0", "--register", "--out", str(tmp_path / 'artifacts')]
    monkeypatch.setattr(sys, 'argv', argv)
    tp.main()

    assert db_path.exists()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT run_id FROM runs")
    assert cur.fetchone() is not None
    cur.execute("SELECT path FROM artifacts")
    assert cur.fetchone() is not None
    conn.close()
