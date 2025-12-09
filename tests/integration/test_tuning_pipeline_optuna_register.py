import os
import json
import sqlite3
import pandas as pd
import sys
import pytest
from importlib import reload

from orchestration.pipelines import tuning_pipeline as tp


def test_tuning_pipeline_optuna_register(tmp_path, monkeypatch):
    pytest.importorskip('optuna')
    # prepare CSV
    idx = pd.date_range('2021-01-01', periods=120, freq='1min', tz='UTC')
    df = pd.DataFrame({'timestamp': idx, 'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': (100 + pd.Series(range(120))/10.0), 'volume': 1.0})
    csv_path = tmp_path / 'prices.csv'
    df.to_csv(csv_path, index=False)

    db_path = tmp_path / 'registry.db'
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv('EXPERIMENT_DB_URL', db_url)

    param_space = json.dumps({'short': [5], 'long': [10], 'size': [1.0]})
    argv = ["tuning_pipeline.py", "--prices-csv", str(csv_path), "--n-trials", "2", "--optimizer", "optuna", "--param-space", param_space, "--register", "--output", str(tmp_path / 'artifacts')]
    monkeypatch.setattr(sys, 'argv', argv)
    tp.main()

    # verify DB entries
    assert db_path.exists()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'")
    assert cur.fetchone() is not None
    cur.execute("SELECT run_id FROM runs")
    assert cur.fetchone() is not None
    conn.close()
