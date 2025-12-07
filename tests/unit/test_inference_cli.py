import json
import pickle
import numpy as np
import pandas as pd

from src.inference.cli import run_prediction_from_weights


def test_inference_cli_ml(tmp_path):
    # create a simple coef-based checkpoint
    ck = {'model': {'coef': [2.0, 3.0]}}
    ckpt = tmp_path / 'm.pkl'
    with open(ckpt, 'wb') as fh:
        pickle.dump(ck, fh)

    # weights JSON mapping low -> this checkpoint
    w = {'model_names': ['m'], 'weights': {'m': 1.0}, 'checkpoints': {'m': str(ckpt)}, 'regime_map': {'low': str(ckpt)}}
    wpath = tmp_path / 'weights.json'
    wpath.write_text(json.dumps(w))

    # input row: [1, 2] -> prediction = 2*1 + 3*2 = 8
    out = run_prediction_from_weights(str(wpath), 'low', [1.0, 2.0])
    assert np.allclose(out, 8.0)
