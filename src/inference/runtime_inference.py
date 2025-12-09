from __future__ import annotations

import os
import pickle
from typing import Any, Callable, Optional

from src.models.runtime_selector import load_ensemble_weights, get_checkpoint_for_regime


def select_checkpoint_for_current_regime(weights_json: str, regime: str) -> Optional[str]:
    """Return absolute checkpoint path for the given regime using the ensemble weights JSON."""
    obj = load_ensemble_weights(weights_json)
    ckpt = get_checkpoint_for_regime(obj, regime)
    if ckpt:
        return os.path.abspath(ckpt)
    return None


def load_ml_checkpoint(checkpoint_path: str) -> Callable[[Any], Any]:
    """Load a pickle-based ML checkpoint and return a predict(callable).

    The returned callable accepts a 2D array-like `X` and returns predictions.
    The checkpoint is expected to be a pickled dict similar to `train_ml` output
    where `model` may contain a `sklearn_model` with `.predict()` or a numpy `coef`.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    with open(checkpoint_path, 'rb') as fh:
        ck = pickle.load(fh)

    model_info = ck.get('model') if isinstance(ck, dict) else None

    def _predict(X):
        import numpy as _np

        if model_info is None:
            raise RuntimeError('checkpoint format not recognized')
        if 'sklearn_model' in model_info:
            return model_info['sklearn_model'].predict(X)
        coef = model_info.get('coef')
        if coef is not None:
            coef = _np.asarray(coef).reshape(-1)
            Xa = _np.asarray(X)
            return Xa.dot(coef)
        raise RuntimeError('no supported model inside checkpoint')

    return _predict


def load_rl_policy(checkpoint_path: str, obs_dim: int | None = None, act_dim: int | None = None):
    """Load a PPO checkpoint into a `PPOTrainer` actor and return a callable that maps obs -> action.

    If `obs_dim`/`act_dim` are None, attempt to read `<checkpoint>.meta.json` for metadata.
    Returns a function taking a single observation (1D array-like) and returning an action numpy array.
    """
    try:
        from src.models.rl.ppo import PPOTrainer, PPOConfig
    except Exception as exc:
        raise RuntimeError('PPOTrainer not available; ensure PyTorch and rl module are installed') from exc

    # attempt to auto-detect dims from metadata if not provided
    if obs_dim is None or act_dim is None:
        meta_path = os.path.splitext(checkpoint_path)[0] + '.meta.json'
        if os.path.exists(meta_path):
            try:
                import json as _json

                with open(meta_path, 'r', encoding='utf-8') as mf:
                    meta = _json.load(mf)
                if obs_dim is None and 'obs_dim' in meta:
                    obs_dim = int(meta.get('obs_dim'))
                if act_dim is None and 'act_dim' in meta:
                    act_dim = int(meta.get('act_dim'))
            except Exception:
                pass

    if obs_dim is None or act_dim is None:
        raise ValueError('obs_dim and act_dim must be provided or available in checkpoint metadata')

    trainer = PPOTrainer(obs_dim=obs_dim, act_dim=act_dim, cfg=PPOConfig())
    try:
        trainer.load(checkpoint_path)
    except Exception as exc:
        raise RuntimeError(f'failed to load RL checkpoint: {checkpoint_path}') from exc

    def _act(obs):
        import numpy as _np
        import torch as _torch

        arr = _np.asarray(obs, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        t = _torch.as_tensor(arr, dtype=_torch.float32, device=trainer.device)
        with _torch.no_grad():
            mu, _ = trainer.actor(t)
        out = mu.cpu().numpy()
        return out.squeeze(0)

    return _act


__all__ = ['select_checkpoint_for_current_regime', 'load_ml_checkpoint']
