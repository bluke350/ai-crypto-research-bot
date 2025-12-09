"""Inspect actor outputs for a checkpoint on the first observation of a pair."""
from __future__ import annotations
import sys
from src.models.rl.data import load_price_history
from src.models.rl.replay_env import ReplayEnv
from src.models.rl.ppo import PPOTrainer
import numpy as np

def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else 'models/ppo_sim_2k.pth'
    pair = sys.argv[2] if len(sys.argv) > 2 else 'XBT/USD'
    prices = load_price_history('data/raw', pair)
    env = ReplayEnv(prices)
    obs = env.reset()
    trainer = PPOTrainer(obs_dim=obs.shape[0], act_dim=1)
    trainer.load(ckpt)
    import torch
    obs_t = torch.as_tensor(np.array(obs, dtype=np.float32).reshape(1, -1))
    mu, std = trainer.actor(obs_t)
    print('actor mu:', mu.detach().cpu().numpy(), 'std:', std.detach().cpu().numpy())
    print('actor mu*5:', (mu*5).detach().cpu().numpy(), 'std*5:', (std*5).detach().cpu().numpy())

if __name__ == '__main__':
    main()
