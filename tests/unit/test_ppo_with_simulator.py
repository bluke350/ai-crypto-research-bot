import importlib
import pytest


def _skip_if_no_torch():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed - skipping PPO+Simulator integration test")


def test_ppo_with_simulator_env_small_run():
    _skip_if_no_torch()
    from src.models.rl.env import SimulatorEnv
    from src.models.rl.ppo import PPOTrainer, PPOConfig

    env = SimulatorEnv(init_price=100.0, init_cash=10000.0, max_steps=50)
    obs = env.reset()
    obs_dim = obs.shape[0]
    trainer = PPOTrainer(obs_dim=obs_dim, act_dim=1, cfg=PPOConfig(epochs=1, minibatch_size=8))
    metrics = trainer.train(env, total_steps=80, rollout_len=40)
    assert metrics.get("total_steps", 0) >= 80
