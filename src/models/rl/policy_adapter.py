from __future__ import annotations

from typing import Optional
import os

import pandas as pd

try:
    import torch
    from src.models.rl.ppo import Actor
except Exception as exc:
    # keep import-time failure explicit for environments without torch
    raise


class RLPolicyStrategy:
    """Adapter that loads a trained PPO actor checkpoint and exposes
    generate_targets(prices) -> pd.Series of target units aligned to prices.

    Notes:
    - The adapter infers the actor's input dimension from the checkpoint and
      constructs an Actor model, then loads the state_dict.
    - Observations are constructed as recent returns (or zeros if insufficient history).
    - Action returned by the actor is passed through tanh and scaled by `size`.
    """

    def __init__(self, ckpt_path: str, size: float = 1.0, obs_window: Optional[int] = None, device: Optional[str] = None):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        self.ckpt_path = ckpt_path
        self.size = float(size)
        # obs_window may be None: infer from checkpoint below when possible
        self.obs_window = int(obs_window) if obs_window is not None else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        data = torch.load(ckpt_path, map_location="cpu")
        # state dict keys for Actor are like 'net.0.weight' etc. infer input dim
        actor_sd = data.get("actor") if isinstance(data, dict) else None
        if actor_sd is None:
            # assume whole checkpoint is actor state dict
            actor_sd = data

        # find a weight key representing the first linear layer
        in_dim = None
        for k, v in actor_sd.items():
            if k.endswith("net.0.weight"):
                in_dim = v.shape[1]
                break
        if in_dim is None:
            # fallback: try to examine any weight shape
            for v in actor_sd.values():
                if hasattr(v, "shape") and v.ndim == 2:
                    in_dim = v.shape[1]
                    break
        if in_dim is None:
            raise RuntimeError("cannot infer actor input dimension from checkpoint")

        # actor output is expected to be scalar action (act_dim=1)
        self.actor = Actor(obs_dim=int(in_dim), act_dim=1).to(self.device)
        self.actor.load_state_dict(actor_sd)
        self.actor.eval()

        # If obs_window was not provided by the caller, adopt the checkpoint's input dim.
        # If a caller provided an obs_window, ensure it matches the checkpoint input
        # dimension; raise a clear error if incompatible to avoid silent shape mismatches.
        if self.obs_window is None:
            self.obs_window = int(in_dim)
        else:
            try:
                if int(self.obs_window) != int(in_dim):
                    raise ValueError(
                        f"Provided obs_window={self.obs_window} incompatible with checkpoint input dim={in_dim}"
                    )
            except Exception:
                # if casting failed, raise a ValueError for clarity
                raise ValueError("Invalid obs_window provided; must be an integer compatible with model input dim")

    def _build_obs(self, recent_closes: pd.Series) -> torch.Tensor:
        import numpy as _np
        import torch as _torch

        # build returns vector of length obs_window; pad left with zeros if needed
        if recent_closes.empty:
            arr = _np.zeros(self.obs_window, dtype=_np.float32)
        else:
            # compute simple returns
            r = recent_closes.pct_change().fillna(0.0).to_numpy(dtype=_np.float32)
            if len(r) >= self.obs_window:
                arr = r[-self.obs_window:]
            else:
                pad = _np.zeros(self.obs_window - len(r), dtype=_np.float32)
                arr = _np.concatenate([pad, r])

        return _torch.as_tensor(arr.reshape(1, -1), dtype=_torch.float32, device=self.device)

    def generate_targets(self, prices: pd.DataFrame) -> pd.Series:
        """Return target units (float) aligned to prices.index.

        Expects a DataFrame with a 'close' column and a DatetimeIndex or timestamp column.
        """
        import torch as _torch

        df = prices.copy()
        if "close" not in df.columns:
            raise ValueError("prices must contain 'close' column")
        closes = df["close"].astype(float)
        out = []
        for i in range(len(closes)):
            start = max(0, i - self.obs_window + 1)
            recent = closes.iloc[start : i + 1]
            obs = self._build_obs(recent)
            with _torch.no_grad():
                mu, _std = self.actor(obs)
                action = _torch.tanh(mu).cpu().numpy().squeeze().item()
            out.append(action * self.size)

        series = pd.Series(out, index=df.index, name="target")
        return series
