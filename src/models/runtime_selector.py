from __future__ import annotations

import json
import os
from typing import Optional, Dict


def load_ensemble_weights(weights_json: str) -> Dict:
    """Load ensemble weights JSON created by the training pipeline."""
    if not os.path.exists(weights_json):
        raise FileNotFoundError(weights_json)
    with open(weights_json, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def get_checkpoint_for_regime(weights_obj: Dict, regime: str) -> Optional[str]:
    """Return checkpoint path for given regime using 'regime_map' then 'checkpoints' fallbacks.

    weights_obj: parsed JSON containing keys: 'regime_map', 'checkpoints'
    """
    if not weights_obj:
        return None
    # prefer regime_map if present
    regime_map = weights_obj.get('regime_map') or {}
    if regime in regime_map and regime_map[regime]:
        return regime_map[regime]
    # fallback: pick the checkpoint with highest weight from 'checkpoints' mapping
    checkpoints = weights_obj.get('checkpoints') or {}
    weights = weights_obj.get('weights') or {}
    # if checkpoints contain regime substring, prefer that
    for k, v in checkpoints.items():
        if k and regime in k and v:
            return v
    # otherwise choose model with highest weight that has a checkpoint
    if weights and checkpoints:
        sorted_models = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
        for m, _ in sorted_models:
            if checkpoints.get(m):
                return checkpoints.get(m)
    # last resort: return any checkpoint path
    for v in checkpoints.values():
        if v:
            return v
    return None


__all__ = ['load_ensemble_weights', 'get_checkpoint_for_regime']
