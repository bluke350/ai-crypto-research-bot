import pickle
import json
import numpy as np

from src.inference.runtime_inference import select_checkpoint_for_current_regime, load_ml_checkpoint


def test_runtime_inference_ml(tmp_path):
    # create a fake ML checkpoint (pickle) with sklearn-like predict
    # Use a simple coef-based model to avoid pickling local classes
    ck = {
        'model': {'coef': [1.0, 1.0]},
    }
    ckpt = tmp_path / 'modelA.pkl'
    with open(ckpt, 'wb') as fh:
        pickle.dump(ck, fh)

    # create weights json mapping regime->checkpoint
    w = {'model_names': ['modelA'], 'weights': {'modelA': 1.0}, 'checkpoints': {'modelA': str(ckpt)}, 'regime_map': {'low': str(ckpt)}}
    wpath = tmp_path / 'weights.json'
    wpath.write_text(json.dumps(w))

    sel = select_checkpoint_for_current_regime(str(wpath), 'low')
    assert sel is not None
    pred_fn = load_ml_checkpoint(sel)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = pred_fn(X)
    assert np.allclose(out, np.array([3.0, 7.0]))
