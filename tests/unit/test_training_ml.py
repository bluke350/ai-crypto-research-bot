from __future__ import annotations

import sys
import os

from orchestration.pipelines import training_pipeline


def test_training_pipeline_ml(tmp_path, monkeypatch):
    save_path = tmp_path / "ml_test_checkpoint.pkl"
    prev_argv = list(sys.argv)
    try:
        sys.argv = [
            "training_pipeline.py",
            "--model",
            "ml",
            "--save",
            str(save_path),
            "--steps",
            "5",
            "--seed",
            "0",
        ]
        # run main; it should create the save file
        training_pipeline.main()
    finally:
        sys.argv = prev_argv

    assert os.path.exists(save_path)
