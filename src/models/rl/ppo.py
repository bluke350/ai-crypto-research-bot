"""Lightweight PPO trainer (PyTorch).

This file provides a minimal, easy-to-read PPO trainer suitable for research
and local experiments. It expects an environment with a gym-like interface
(`reset() -> obs`, `step(action) -> (obs, reward, done, info)`) and assumes
continuous action spaces (Box). The trainer is intentionally simple and
documented; swap in a more advanced implementation (stable-baselines3, rllib)
in production.

Design contract (inputs/outputs):
- Input: env implementing reset/step, observation shape, action shape
- Output: trained policy checkpoints saved via `save()` and returned metrics

This implementation is small and adds: policy/value MLPs, GAE advantage,
PPO clipped objective, and a `train()` loop that collects rollouts and
performs updates.

Note: This module requires PyTorch. If unavailable, importing will raise.
"""
from __future__ import annotations

import dataclasses
import math
import os
import logging
from typing import Any, Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
except Exception as exc:  # pragma: no cover - exercised in environments with torch
    raise ImportError("pytorch is required for src.models.rl.ppo but is not available") from exc


def _build_mlp(in_dim: int, out_dim: int, hidden_sizes=(64, 64), activation=nn.Tanh):
    layers = []
    prev = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(64, 64)):
        super().__init__()
        self.net = _build_mlp(obs_dim, act_dim, hidden_sizes)
        # global log std parameter (state-independent)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        # optional action scale, trainer may set attribute `action_scale` on this module
        self.action_scale = 1.0

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.net(obs)
        # apply optional action scaling (keeps sampling and update consistent)
        mu = mu * getattr(self, "action_scale", 1.0)
        std = torch.exp(self.log_std)
        return mu, std


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes=(64, 64)):
        super().__init__()
        self.net = _build_mlp(obs_dim, 1, hidden_sizes)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


@dataclasses.dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_epsilon: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    action_scale: float = 1.0
    lr: float = 3e-4
    epochs: int = 10
    batch_size: int = 64
    minibatch_size: int = 32
    max_grad_norm: float = 0.5
    verbose: bool = False
    log_interval: int = 1


class PPOTrainer:
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig = None, device: Optional[str] = None):
        self.cfg = cfg or PPOConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # store dims for metadata and possible runtime inspection
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        # propagate action scale to actor module so forward() applies scaling
        try:
            self.actor.action_scale = float(self.cfg.action_scale)
        except Exception:
            self.actor.action_scale = 1.0
        self.critic = Critic(obs_dim).to(self.device)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.cfg.lr)
        self.logger = logging.getLogger(__name__)

    def _sample_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, std = self.actor(obs)
        dist = Normal(mu, std)
        act = dist.sample()
        logp = dist.log_prob(act).sum(axis=-1)
        return act, logp, mu

    def _compute_gae(self, rewards, masks, values):
        # rewards, masks, values: tensors (T,)
        returns = torch.zeros_like(rewards)
        adv = torch.zeros_like(rewards)
        last_gae = 0.0
        next_value = 0.0
        for t in reversed(range(rewards.shape[0])):
            delta = rewards[t] + self.cfg.gamma * next_value * masks[t] - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.lam * masks[t] * last_gae
            adv[t] = last_gae
            next_value = values[t]
        returns = adv + values
        return adv, returns

    def update(self, obs_buf, act_buf, logp_buf, ret_buf, adv_buf):
        import numpy as _np

        # convert lists-of-arrays into proper numpy arrays to avoid slow tensor creation warnings
        obs_arr = _np.asarray(obs_buf, dtype=_np.float32)
        acts_arr = _np.asarray(act_buf, dtype=_np.float32)
        old_logp_arr = _np.asarray(logp_buf, dtype=_np.float32)
        ret_arr = _np.asarray(ret_buf, dtype=_np.float32)
        adv_arr = _np.asarray(adv_buf, dtype=_np.float32)

        obs = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(acts_arr, dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(old_logp_arr, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(ret_arr, dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(adv_arr, dtype=torch.float32, device=self.device)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        dataset = torch.utils.data.TensorDataset(obs, acts, old_logp, ret, adv)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg.minibatch_size, shuffle=True)

        for _ in range(self.cfg.epochs):
            for b_obs, b_acts, b_old_logp, b_ret, b_adv in loader:
                mu, std = self.actor(b_obs)
                dist = Normal(mu, std)
                new_logp = dist.log_prob(b_acts).sum(axis=-1)
                ratio = torch.exp(new_logp - b_old_logp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value = self.critic(b_obs)
                value_loss = (b_ret - value).pow(2).mean()
                entropy = dist.entropy().sum(axis=-1).mean()

                loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.cfg.max_grad_norm)
                self.optimizer.step()

        return {
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "entropy": float(entropy.detach().cpu().item()),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
        # write small metadata file next to checkpoint to allow auto-detection
        try:
            import json
            from src.utils.time import now_iso

            meta = {
                "checkpoint": os.path.abspath(path),
                "created_at": now_iso() + 'Z',
                "obs_dim": int(getattr(self, 'obs_dim', -1)),
                "act_dim": int(getattr(self, 'act_dim', -1)),
                "cfg": {k: getattr(self.cfg, k) for k in self.cfg.__dict__ if not k.startswith('_')},
            }
            meta_path = os.path.splitext(path)[0] + ".meta.json"
            with open(meta_path, 'w', encoding='utf-8') as mf:
                json.dump(meta, mf)
        except Exception:
            # metadata writing should not break training
            pass

    def load(self, path: str, map_location: Optional[str] = None):
        data = torch.load(path, map_location=map_location or self.device)
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.optimizer.load_state_dict(data.get("optimizer", {}))

    def train(self, env, total_steps: int = 10_000, rollout_len: int = 2048, save_path: Optional[str] = None):
        """Main training loop.

        env: gym-like environment with Box action space and np.ndarray observations
        total_steps: total environment steps to collect
        rollout_len: steps per trajectory batch
        """
        import numpy as np

        obs = env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        step_count = 0
        metrics = {"episodes": 0, "total_steps": 0}

        while step_count < total_steps:
            obs_buf = []
            act_buf = []
            rew_buf = []
            mask_buf = []
            logp_buf = []
            val_buf = []

            for _ in range(rollout_len):
                with torch.no_grad():
                    a, logp, _ = self._sample_action(obs.unsqueeze(0))
                a_np = a.cpu().numpy().squeeze(0)
                next_obs, r, done, _ = env.step(a_np)

                # store
                obs_buf.append(obs.cpu().numpy())
                act_buf.append(a_np)
                logp_buf.append(logp.cpu().numpy())
                rew_buf.append(float(r))
                mask_buf.append(0.0 if done else 1.0)

                with torch.no_grad():
                    val = self.critic(obs.unsqueeze(0)).cpu().numpy().squeeze(0)
                val_buf.append(float(val))

                step_count += 1
                metrics["total_steps"] = step_count

                if done:
                    metrics["episodes"] += 1
                    next_obs = env.reset()

                obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
                if step_count >= total_steps:
                    break

            # compute last value for bootstrap
            with torch.no_grad():
                last_val = float(self.critic(obs.unsqueeze(0)).cpu().numpy().squeeze(0))

            # append last bootstrap value for GAE algorithm
            import numpy as _np

            rew_arr = _np.array(rew_buf, dtype=_np.float32)
            mask_arr = _np.array(mask_buf, dtype=_np.float32)
            val_arr = _np.array(val_buf + [last_val], dtype=_np.float32)

            # compute advantages
            adv = _np.zeros_like(rew_arr)
            lastgaelam = 0
            for t in reversed(range(len(rew_arr))):
                nextnonterminal = mask_arr[t]
                delta = rew_arr[t] + self.cfg.gamma * val_arr[t + 1] * nextnonterminal - val_arr[t]
                lastgaelam = delta + self.cfg.gamma * self.cfg.lam * nextnonterminal * lastgaelam
                adv[t] = lastgaelam
            returns = adv + val_arr[:-1]

            # update
            stats = self.update(obs_buf, act_buf, logp_buf, returns, adv)

            # compute simple rollout return for monitoring
            rollout_return = float(sum(rew_buf)) if rew_buf else 0.0

            # periodic logging: print per-update stats and rollout return when verbose
            if self.cfg.verbose:
                self.logger.info("PPO update steps=%s episodes=%s rollout_return=%s stats=%s",
                                 metrics["total_steps"], metrics["episodes"], rollout_return, stats)
                print(f"PPO update steps={metrics['total_steps']} episodes={metrics['episodes']} rollout_return={rollout_return:.6f} stats={stats}")

            if save_path:
                self.save(save_path)

        return metrics
