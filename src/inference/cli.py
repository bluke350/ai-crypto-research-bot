from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

from .runtime_inference import select_checkpoint_for_current_regime, load_ml_checkpoint, load_rl_policy


def run_prediction_from_weights(weights_json: str, regime: str, input_row: Sequence[float], rl_obs_dim: int = None, rl_act_dim: int = None):
    """Select checkpoint for `regime` and run a single prediction using `input_row`.

    - For ML checkpoints (pickle), returns the scalar or array prediction.
    - For RL checkpoints (.pt/.pth), requires `rl_obs_dim` and `rl_act_dim` and returns the action array.
    """
    ckpt = select_checkpoint_for_current_regime(weights_json, regime)
    if not ckpt:
        raise FileNotFoundError(f"no checkpoint found for regime {regime}")

    # infer type by extension
    ext = os.path.splitext(ckpt)[1].lower()
    if ext in ('.pkl', '.pickle'):
        pred_fn = load_ml_checkpoint(ckpt)
        return pred_fn([input_row])[0] if hasattr(pred_fn([input_row]), '__len__') else pred_fn([input_row])
    else:
        # treat as RL checkpoint
        if rl_obs_dim is None or rl_act_dim is None:
            raise ValueError('rl_obs_dim and rl_act_dim are required for RL policy loading')
        act_fn = load_rl_policy(ckpt, obs_dim=rl_obs_dim, act_dim=rl_act_dim)
        return act_fn(input_row)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--weights-json', type=str, required=True)
    p.add_argument('--regime', type=str, required=True)
    p.add_argument('--input-csv', type=str, default=None, help='CSV with a single row of features (first row used)')
    p.add_argument('--rl-obs-dim', type=int, default=None)
    p.add_argument('--rl-act-dim', type=int, default=None)
    args = p.parse_args(argv)

    if args.input_csv:
        import pandas as pd
        from src.utils.io import load_prices_csv
        df = load_prices_csv(args.input_csv, dedupe='first')
        if df.shape[0] == 0:
            raise ValueError('input CSV is empty')
        row = df.iloc[0].values.tolist()
    else:
        raise ValueError('input_csv is required for CLI usage')

    out = run_prediction_from_weights(args.weights_json, args.regime, row, rl_obs_dim=args.rl_obs_dim, rl_act_dim=args.rl_act_dim)
    # print JSON-serializable result
    try:
        print(json.dumps({'prediction': out.tolist() if hasattr(out, 'tolist') else out}))
    except Exception:
        print({'prediction': out})


if __name__ == '__main__':
    main()
