from __future__ import annotations

import os
import pickle
import tempfile

import pandas as pd

from orchestration.paper_runner import run_paper


def _make_ckpt(path):
    ckpt = {"model": {"coef": [0.1, 0.0]}, "type": "linear"}
    with open(path, "wb") as fh:
        pickle.dump(ckpt, fh)


def test_paper_runner_with_synthetic_prices(tmp_path):
    ckpt = tmp_path / "ckpt.pkl"
    _make_ckpt(str(ckpt))
    out_dir = run_paper(str(ckpt), prices_csv=None, out_root=str(tmp_path))
    assert os.path.exists(out_dir)
    assert os.path.exists(os.path.join(out_dir, "result.json"))
    assert os.path.exists(os.path.join(out_dir, "summary.json"))
