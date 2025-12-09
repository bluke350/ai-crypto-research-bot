import os
import sys
import json
import uuid

import pytest


def test_training_pipeline_writes_ensemble_weights(tmp_path, monkeypatch):
    # monkeypatch a minimal ml trainer that writes a fake checkpoint and returns its path
    fake_model_name = "foo"
    out = tmp_path / "out"
    out.mkdir()

    def fake_train_ml(data_root, save, steps, seed):
        # create the save file
        p = tmp_path / f"{fake_model_name}_ckpt_seed{seed}.pth"
        p.write_text("fake")
        return str(p)

    # ensure the import path exists for trainer
    monkeypatch.setitem(sys.modules, 'src.training.trainer', type('M', (), {'train_ml': fake_train_ml}))

    # call the training pipeline main with args
    from orchestration.pipelines import training_pipeline

    prev_argv = sys.argv
    try:
        # also create an explicit mapping file for 'bar' to exercise --ensemble-map
        bar_file = tmp_path / "bar_external.pth"
        bar_file.write_text("external")
        sys.argv = [
            "train", "--model", "ml",
            "--save", str(tmp_path / 'unused.pth'),
            "--out", str(out),
            "--ensemble-names", f"{fake_model_name},bar",
            "--ensemble-map", f"bar={str(bar_file)}",
            "--seed", "7",
        ]
        # run main (should create out/<run_id>/ensemble_weights.json)
        training_pipeline.main()
    finally:
        sys.argv = prev_argv

    # locate the generated run directory
    run_dirs = list(out.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    weights_file = run_dir / 'ensemble_weights.json'
    assert weights_file.exists(), f"weights file not found in {run_dir}"
    data = json.loads(weights_file.read_text(encoding='utf-8'))
    assert 'model_names' in data and isinstance(data['model_names'], list)
    assert fake_model_name in data['model_names']
    assert 'weights' in data and isinstance(data['weights'], dict)
    assert 'checkpoints' in data and isinstance(data['checkpoints'], dict)
    # checkpoint mapping for foo should point to the fake file
    ckpt = data['checkpoints'].get(fake_model_name)
    assert ckpt is not None and os.path.exists(ckpt)
