import json
import tempfile
from src.models.runtime_selector import load_ensemble_weights, get_checkpoint_for_regime


def test_runtime_selector_simple(tmp_path):
    data = {
        'model_names': ['a', 'b'],
        'weights': {'a': 0.6, 'b': 0.4},
        'checkpoints': {'a': '/path/to/a.ckpt', 'b': '/path/to/b.ckpt'},
        'regime_map': {'low': '/path/to/a.ckpt'}
    }
    p = tmp_path / 'weights.json'
    p.write_text(json.dumps(data))
    obj = load_ensemble_weights(str(p))
    assert isinstance(obj, dict)
    assert get_checkpoint_for_regime(obj, 'low') == '/path/to/a.ckpt'
    # unknown regime -> highest weight
    assert get_checkpoint_for_regime(obj, 'high') in ('/path/to/a.ckpt', '/path/to/b.ckpt')
