"""Evaluate a saved PPO policy on historical data and output NAV time series.

Usage:
  python -m src.models.rl.evaluate_policy --ckpt models/ppo_smoke.pth --pair "XBT/USD" --data-root data/raw --out results/eval_nav.csv
"""
from __future__ import annotations

import argparse
import os
import pandas as pd

from src.models.rl.data import load_price_history
from src.models.rl.replay_env import ReplayEnv

def run_eval(ckpt: str, data_root: str, pair: str, out: str, stochastic_runs: int = 1, action_scale: float = 1.0, action_mode: str = "absolute"):
    """Run N stochastic evaluations and average NAVs.

    Parameters
    - stochastic_runs: number of stochastic rollouts to average
    - action_scale: multiply policy outputs (mu/std) by this factor before applying
    - action_mode: 'absolute' or 'relative' interpretation of actions
    """
    from src.models.rl.ppo import PPOTrainer
    from src.models.rl.wrappers import RelativeActionWrapper

    prices = load_price_history(data_root, pair)
    if prices.empty:
        raise RuntimeError(f"no price data found for {pair} under {data_root}")

    base_env = ReplayEnv(prices)

    import torch
    import numpy as _np
    from torch.distributions import Normal

    obs = base_env.reset()
    obs_dim = obs.shape[0]
    trainer = PPOTrainer(obs_dim=obs_dim, act_dim=1)
    trainer.load(ckpt)

    # if action_mode is relative, we'll wrap each env instance used for rollout
    use_relative = action_mode == "relative"

    all_navs = []
    timestamps = None

    for run_idx in range(max(1, int(stochastic_runs))):
        # fresh env per run
        env = ReplayEnv(prices)
        if use_relative:
            env = RelativeActionWrapper(env)
        # observation normalization
        if action_mode is None:
            action_mode = "absolute"
        # note: action_mode handled above; obs_mode will be passed through via caller
        obs_mode = getattr(run_eval, "_obs_mode", "raw")
        if obs_mode != "raw":
            from src.models.rl.wrappers import NormalizedObservationWrapper
            env = NormalizedObservationWrapper(env, mode=obs_mode)

        obs = env.reset()

        nav = []
        # initial nav
        price0 = float(prices.iloc[0]["close"])
        nav_val = env.cash + getattr(env, "position", 0.0) * price0
        nav.append(nav_val)

        done = False
        while not done:
            obs_t = torch.as_tensor(_np.array(obs, dtype=_np.float32).reshape(1, -1))
            with torch.no_grad():
                mu, std = trainer.actor(obs_t.to(trainer.device))
            # apply action scaling requested at eval time
            mu = mu * float(action_scale)
            std = std * float(action_scale)

            # always stochastic sampling for ensemble diversity; for deterministic single-run pass stochastic_runs=1 and action_scale as needed
            dist = Normal(mu, std)
            a = dist.sample()
            act = a.cpu().numpy().squeeze(0)
            obs, reward, done, info = env.step(act)
            nav_val += float(reward)
            # try to capture timestamp aligned with env index
            idx = max(0, getattr(env, "idx", getattr(env, "step_count", 0)) - 1)
            ts = prices.iloc[idx]["timestamp"]
            nav.append(nav_val)

        all_navs.append(nav)
        # record timestamps from the first run
        if timestamps is None:
            # first entry was initial price
            timestamps = [prices.iloc[0]["timestamp"]] + [prices.iloc[max(0, i - 1)]["timestamp"] for i in range(1, len(nav))]

    # align and average navs (runs should be same length)
    import math
    lens = [len(x) for x in all_navs]
    if len(set(lens)) != 1:
        # pad shorter runs with last value
        maxlen = max(lens)
        for i, x in enumerate(all_navs):
            if len(x) < maxlen:
                all_navs[i] = x + [x[-1]] * (maxlen - len(x))

    arr = _np.array(all_navs, dtype=_np.float64)
    mean_nav = arr.mean(axis=0)

    df = pd.DataFrame({"timestamp": timestamps, "nav": mean_nav})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print("wrote", out)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--pair", required=True)
    p.add_argument("--out", default="results/eval_nav.csv")
    p.add_argument("--stochastic-runs", type=int, default=1, help="Number of stochastic runs to average (ensemble). If >1, samples from policy distribution.")
    p.add_argument("--action-scale", type=float, default=1.0, help="Scale factor to multiply policy outputs (mu/std) at eval time")
    p.add_argument("--action-mode", type=str, default="absolute", choices=("absolute", "relative"), help="Interpret actions as absolute desired position or relative delta")
    p.add_argument("--obs-mode", type=str, default="raw", choices=("raw", "returns", "logdiff"), help="Observation preprocessing mode to apply at eval time")
    args = p.parse_args()
    # pass obs_mode into run_eval via attribute so inner loop can pick it up
    setattr(run_eval, "_obs_mode", getattr(args, "obs_mode", "raw"))
    run_eval(args.ckpt, args.data_root, args.pair, args.out, stochastic_runs=args.stochastic_runs, action_scale=args.action_scale, action_mode=args.action_mode)


if __name__ == "__main__":
    main()
