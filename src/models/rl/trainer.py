import numpy as np
from .env import SimpleTradingEnv


def random_policy(env, episodes=10):
    results = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        results.append(total_reward)
    return results


def train_random(prices, episodes=10):
    env = SimpleTradingEnv(np.array(prices))
    return random_policy(env, episodes=episodes)
