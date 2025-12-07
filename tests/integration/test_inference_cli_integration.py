import json
import pickle
import sys
import subprocess
from pathlib import Path


def test_inference_cli_with_pickled_ml_model(tmp_path):
    # create a tiny checkpoint with linear coef so no custom class is required
    # model will compute dot(X, coef) -> [1,1] dot [1,2] == 3
    model_obj = {'model': {'coef': [1.0, 1.0]}}

    model_path = tmp_path / 'dummy_model.pkl'
    with open(model_path, 'wb') as fh:
        pickle.dump(model_obj, fh)

    weights = {'regime_map': {'mid': str(model_path)}}
    weights_path = tmp_path / 'weights.json'
    with open(weights_path, 'w', encoding='utf-8') as fh:
        json.dump(weights, fh)

    # create input CSV with a single row of features
    input_csv = tmp_path / 'input.csv'
    input_csv.write_text('f1,f2\n1,2\n')

    # run the CLI as a module
    cmd = [sys.executable, '-m', 'src.inference.cli', '--weights-json', str(weights_path), '--regime', 'mid', '--input-csv', str(input_csv)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = proc.stdout.strip()
    # should be a JSON line with prediction 3.0
    parsed = json.loads(out)
    assert 'prediction' in parsed
    # DummyModel sums inputs [1,2] -> 3.0
    assert parsed['prediction'] == 3.0
