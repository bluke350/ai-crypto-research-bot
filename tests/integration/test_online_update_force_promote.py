import os
import subprocess
from pathlib import Path


def test_online_update_force_promote(tmp_path, monkeypatch):
    repo_root = Path.cwd()
    # Ensure models dir
    models = repo_root / 'models'
    models.mkdir(exist_ok=True)
    # Train fallback model
    subprocess.run(['python', 'tooling/train_fallback_opportunity.py'], check=True)
    # populate buffer
    subprocess.run(['python', '-c', "import pandas as pd; from src.data.online_buffer import append_rows; df = pd.DataFrame({'f0':[0.1,0.2,0.3],'f1':[1,2,3],'f2':[3,2,1],'label':[0,1,0]}); append_rows(df)"])
    # run online_update with force-promote
    out = subprocess.run(['PYTHONPATH=.', 'python', 'tooling/online_update.py', '--model', 'models/opportunity.pkl', '--out', 'models/opportunity-online-updated.pkl', '--min-rows', '1', '--force-promote'], shell=False)
    assert out.returncode == 0
    assert (models / 'opportunity-online-updated.pkl').exists()
