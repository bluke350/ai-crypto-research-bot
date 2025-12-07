from __future__ import annotations

import os
import pickle

from orchestration.paper_live import run_live


def _make_ckpt(path):
    ckpt = {"model": {"coef": [0.01, 0.0]}, "type": "linear"}
    with open(path, "wb") as fh:
        pickle.dump(ckpt, fh)


def test_paper_live_synthetic(tmp_path):
    ckpt = tmp_path / "ckpt.pkl"
    _make_ckpt(str(ckpt))
    out_dir = run_live(str(ckpt), prices_csv=None, out_root=str(tmp_path), max_ticks=5, stream_delay=0.0, sleep_between=False)
    assert os.path.exists(out_dir)
    assert os.path.exists(os.path.join(out_dir, "result.json"))
    assert os.path.exists(os.path.join(out_dir, "summary.json"))
