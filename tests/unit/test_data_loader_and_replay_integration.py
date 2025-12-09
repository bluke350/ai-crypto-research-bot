import pandas as pd
import numpy as np
import os
from pathlib import Path

from src.models.rl.data import load_price_history
from src.models.rl.replay_env import ReplayEnv


def make_minute_parquet_tree(tmp_path: Path, pair: str = "XBT/USD") -> Path:
    # create directory structure data/raw/<pair>/<YYYYMMDD>/
    base = tmp_path / "data" / "raw" / pair
    base.mkdir(parents=True)
    # create two days of minute files
    for day in ["20250101", "20250102"]:
        day_dir = base / day
        day_dir.mkdir(parents=True, exist_ok=True)
        # create 10 minute parquet files for the day
        t0 = pd.Timestamp("2025-01-01T00:00:00Z") if day == "20250101" else pd.Timestamp("2025-01-02T00:00:00Z")
        for i in range(10):
            ts = (t0 + pd.Timedelta(minutes=i))
            df = pd.DataFrame([{"timestamp": ts, "vwap": 100.0 + i, "volume": 1.0}])
            fname = ts.strftime("%Y%m%dT%H%M") + ".parquet"
            df.to_parquet(day_dir / fname)
    return tmp_path / "data" / "raw"


def test_loader_and_replay(tmp_path: Path):
    data_root = make_minute_parquet_tree(tmp_path)
    pair = "XBT/USD"
    df = load_price_history(str(data_root), pair)
    assert not df.empty
    assert "timestamp" in df.columns and "close" in df.columns

    # build a replay env
    env = ReplayEnv(df, initial_cash=1000.0, seed=1)
    obs = env.reset()
    assert obs.shape == (2,)
    obs, r, done, info = env.step([0.0])
    assert isinstance(r, float)
