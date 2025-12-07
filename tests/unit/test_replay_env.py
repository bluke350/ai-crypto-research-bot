import numpy as np
import pandas as pd
from src.models.rl.replay_env import ReplayEnv


def make_prices(n=50, seed=0):
    rng = np.random.RandomState(seed)
    t = pd.date_range("2021-01-01", periods=n, freq="min")
    p = 100 + np.cumsum(rng.normal(0, 0.1, size=n))
    return pd.DataFrame({"timestamp": t, "close": p})


def test_replay_env_basic():
    prices = make_prices(30)
    env = ReplayEnv(prices, initial_cash=10000.0, seed=123)
    obs = env.reset()
    assert obs.shape == (2,)
    # take a few steps with alternating desired positions
    total_reward = 0.0
    done = False
    for i in range(5):
        action = 1.0 if (i % 2 == 0) else -1.0
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    # at least one step should have been taken
    assert total_reward == total_reward  # trivial sanity check (not NaN)
    assert isinstance(info, dict)
