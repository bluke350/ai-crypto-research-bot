import pandas as pd
import json

from src.utils.io import load_prices_csv


def _make_dup_csv(tmp_path):
    # create a small CSV with duplicate timestamps
    rows = [
        {"timestamp": "2021-01-01T00:00:00Z", "open": 100.0, "close": 101.0, "volume": 10},
        {"timestamp": "2021-01-01T00:00:00Z", "open": 110.0, "close": 111.0, "volume": 20},
        {"timestamp": "2021-01-01T00:01:00Z", "open": 120.0, "close": 121.0, "volume": 30},
    ]
    df = pd.DataFrame(rows)
    p = tmp_path / "dup.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_dedupe_none(tmp_path):
    p = _make_dup_csv(tmp_path)
    df = load_prices_csv(p, dedupe="none")
    # all rows preserved
    assert len(df) == 3


def test_dedupe_first(tmp_path):
    p = _make_dup_csv(tmp_path)
    df = load_prices_csv(p, dedupe="first")
    assert len(df) == 2
    # first duplicate kept (open == 100)
    first_row = df.loc[df["timestamp"] == pd.to_datetime("2021-01-01T00:00:00Z")].iloc[0]
    assert float(first_row["open"]) == 100.0


def test_dedupe_last(tmp_path):
    p = _make_dup_csv(tmp_path)
    df = load_prices_csv(p, dedupe="last")
    assert len(df) == 2
    # last duplicate kept (open == 110)
    first_row = df.loc[df["timestamp"] == pd.to_datetime("2021-01-01T00:00:00Z")].iloc[0]
    assert float(first_row["open"]) == 110.0


def test_dedupe_mean(tmp_path):
    p = _make_dup_csv(tmp_path)
    df = load_prices_csv(p, dedupe="mean")
    # dup aggregated -> 2 rows
    assert len(df) == 2
    row = df.loc[df["timestamp"] == pd.to_datetime("2021-01-01T00:00:00Z")].iloc[0]
    # open should be averaged (100 + 110) / 2 = 105.0
    assert abs(float(row["open"]) - 105.0) < 1e-6
