import subprocess
import os
from pathlib import Path


def test_canary_promote_local():
    repo_root = Path.cwd()
    # train fallback and ensure buffer exists
    subprocess.run(['python', 'tooling/train_fallback_opportunity.py'], check=True)
    subprocess.run(['python', '-c', "import pandas as pd; from src.data.online_buffer import append_rows; df = pd.DataFrame({'f0':[0.1,0.2,0.3],'f1':[1,2,3],'f2':[3,2,1],'label':[0,1,0],'return':[0.01,-0.02,0.005]}); append_rows(df)"])
    # run canary script
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    res = subprocess.run(['python', 'tooling/canary_promote.py'], env=env)
    assert res.returncode == 0
