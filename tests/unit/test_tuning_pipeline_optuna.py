import sys
import os
import json
import tempfile
import pandas as pd
from importlib import reload

from orchestration.pipelines import tuning_pipeline as tp


def test_tuning_pipeline_optuna_smoke(tmp_path, monkeypatch):
    try:
        import optuna  # type: ignore
    except Exception:
        # skip if optuna is not installed in test environment
        return

    # create a small prices csv
    idx = pd.date_range('2021-01-01', periods=120, freq='1min', tz='UTC')
    df = pd.DataFrame({'timestamp': idx, 'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': (100 + pd.Series(range(120))/10.0), 'volume': 1.0})
    csv_path = tmp_path / "prices.csv"
    df.to_csv(csv_path, index=False)

    # setup args
    param_space = {"short": [3], "long": [5], "size": [1.0]}
    argv = ["tuning_pipeline.py", "--prices-csv", str(csv_path), "--param-space", json.dumps(param_space), "--n-trials", "2", "--window", "40", "--step", "10", "--optimizer", "optuna", "--optuna-db", f"sqlite:///{tmp_path / 'optuna.db'}", "--output", str(tmp_path / "artifacts")]
    monkeypatch.setattr(sys, 'argv', argv)

    # call main (should not raise)
    tp.main()

    # verify artifacts created
    out_dir = list((tmp_path / "artifacts").glob('*'))
    assert len(out_dir) == 1
    res_file = list((tmp_path / "artifacts" / out_dir[0].name).glob('results.json'))
    assert len(res_file) == 1
