import os
import json
import pandas as pd
import tempfile
import sys
from importlib import reload

from orchestration.pipelines import auto_train_pipeline as atp


def test_auto_train_pipeline_smoke(tmp_path, monkeypatch):
    idx = pd.date_range('2021-01-01', periods=120, freq='1min', tz='UTC')
    df = pd.DataFrame({'timestamp': idx, 'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': (100 + pd.Series(range(120))/10.0), 'volume': 1.0})
    csv_path = tmp_path / 'prices.csv'
    df.to_csv(csv_path, index=False)

    param_space = {"short": [3], "long": [5], "size": [1.0]}
    args = [ 'auto_train_pipeline.py', '--prices-csv', str(csv_path), '--param-space', json.dumps(param_space), '--n-trials', '2', '--seeds', '0', '--steps', '10', '--output', str(tmp_path / 'out'), '--register' ]
    monkeypatch.setattr(sys, 'argv', args)
    atp.main()

    # check out dir contains eval.json
    out_dirs = list((tmp_path / 'out').glob('*'))
    assert len(out_dirs) == 1
    eval_file = out_dirs[0] / 'eval.json'
    assert eval_file.exists()
    data = json.loads(eval_file.read_text(encoding='utf-8'))
    assert 'models' in data
    assert isinstance(data['models'], list)
