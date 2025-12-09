import importlib
import sys

import pytest


def _skip_if_no_torch():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed - skipping PPO trainer tests")


def make_dummy_env(obs_dim=4, act_dim=1, max_steps=100):
    import numpy as np

    class DummyEnv:
        def __init__(self):
            self.obs_dim = obs_dim
            self.act_dim = act_dim
            self._steps = 0

        def reset(self):
            self._steps = 0
            return np.zeros(self.obs_dim, dtype=float)

        def step(self, action):
            # simple synthetic dynamics: reward is negative L2 of action to encourage small actions
            self._steps += 1
            obs = np.zeros(self.obs_dim, dtype=float)
            reward = -float((action ** 2).sum())
            done = self._steps >= max_steps
            return obs, reward, done, {}

    return DummyEnv()


def test_ppo_trainer_small_run():
    _skip_if_no_torch()
    import numpy as np

    from src.models.rl.ppo import PPOTrainer, PPOConfig

    env = make_dummy_env(obs_dim=3, act_dim=1, max_steps=20)
    cfg = PPOConfig(epochs=2, minibatch_size=8)
    trainer = PPOTrainer(obs_dim=3, act_dim=1, cfg=cfg)
    metrics = trainer.train(env, total_steps=40, rollout_len=20)
    assert metrics.get("total_steps", 0) >= 40
