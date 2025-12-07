"""Small CLI runner to train PPO on the in-repo SimulatorEnv.

This is intentionally tiny: it constructs the env and trainer and runs a
short training loop. Use for quick experiments or as a reference for
integration with real training infra.
"""
from __future__ import annotations

import argparse
import os

try:
    # lazy import so module can be imported in test environments without torch
    from src.models.rl.ppo import PPOTrainer, PPOConfig
    from src.models.rl.env import SimulatorEnv
    from src.models.rl.replay_env import ReplayEnv
    from src.models.rl.data import load_price_history
except Exception:
    PPOTrainer = None  # type: ignore
    SimulatorEnv = None  # type: ignore
    ReplayEnv = None  # type: ignore
    load_price_history = None  # type: ignore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--replay-pair", type=str, default=None, help="Pair to load from data root for replay (e.g. 'XBT/USD')")
    parser.add_argument("--replay-csv", type=str, default=None, help="CSV file to use as replay price history (overrides data_root/replay_pair)")
    parser.add_argument("--data-root", type=str, default=None, help="Root folder for historical data (default: data/raw)")
    parser.add_argument("--save", type=str, default="models/ppo.ckpt")
    parser.add_argument("--ent-coef", type=float, default=None, help="Entropy coefficient override for PPO (float)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose per-update logging")
    parser.add_argument("--action-scale", type=float, default=None, help="Scale factor to multiply policy outputs (mu)")
    parser.add_argument("--action-mode", type=str, default="absolute", choices=("absolute", "relative"), help="Interpret actions as absolute desired position or relative delta")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible runs (affects numpy, random, torch)")
    parser.add_argument("--obs-mode", type=str, default="raw", choices=("raw", "returns", "logdiff"), help="Observation preprocessing mode: raw, returns, or logdiff")
    args = parser.parse_args()

    if PPOTrainer is None or SimulatorEnv is None:
        raise RuntimeError("PPOTrainer or SimulatorEnv not available (missing dependencies)")

    # apply seed for deterministic behaviour when requested
    if getattr(args, "seed", None) is not None:
        import random as _random
        import numpy as _np
        try:
            import torch as _torch
            _torch.manual_seed(int(args.seed))
        except Exception:
            pass
        _random.seed(int(args.seed))
        _np.random.seed(int(args.seed))

    # If --replay is specified, load historical price data and create a ReplayEnv
    # Replay from CSV (explicit) takes precedence over pair-based loading
    if getattr(args, "replay_csv", None):
        import pandas as _pd
        prices = _pd.read_csv(args.replay_csv)
        if 'timestamp' in prices.columns:
            prices['timestamp'] = _pd.to_datetime(prices['timestamp'], utc=True)
        if 'close' not in prices.columns:
            raise RuntimeError("replay CSV must contain 'close' column")
        prices = prices[['timestamp', 'close']]
        env = ReplayEnv(prices, initial_cash=100000.0, seed=args.seed)
    elif getattr(args, "replay_pair", None):
        if load_price_history is None or ReplayEnv is None:
            raise RuntimeError("replay support requires additional modules (pandas/pyarrow)")
        data_root = args.data_root or "data/raw"
        prices = load_price_history(data_root, args.replay_pair)
        if prices.empty:
            raise RuntimeError(f"no price data found for pair {args.replay_pair} under {data_root}")
        env = ReplayEnv(prices, initial_cash=100000.0, seed=args.seed)
    else:
        env = SimulatorEnv(init_price=100.0, init_cash=100000.0, max_steps=200)
    # If requested, wrap env to interpret actions as relative deltas
    if getattr(args, "action_mode", "absolute") == "relative":
        try:
            from src.models.rl.wrappers import RelativeActionWrapper
            env = RelativeActionWrapper(env)
        except Exception:
            raise RuntimeError("failed to import RelativeActionWrapper")
    # If requested, wrap env to normalize observations
    if getattr(args, "obs_mode", "raw") != "raw":
        try:
            from src.models.rl.wrappers import NormalizedObservationWrapper
            env = NormalizedObservationWrapper(env, mode=args.obs_mode)
        except Exception:
            raise RuntimeError("failed to import NormalizedObservationWrapper")
    obs = env.reset()
    obs_dim = obs.shape[0]
    act_dim = 1
    # allow overriding entropy coefficient and verbosity from CLI
    cfg = PPOConfig()
    if args.ent_coef is not None:
        cfg.ent_coef = float(args.ent_coef)
    if args.action_scale is not None:
        cfg.action_scale = float(args.action_scale)
    cfg.verbose = bool(args.verbose)
    trainer = PPOTrainer(obs_dim=obs_dim, act_dim=act_dim, cfg=cfg)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    metrics = trainer.train(env, total_steps=args.steps, rollout_len=256, save_path=args.save)
    # After training, run a short evaluation to collect critic predictions and empirical returns
    try:
        import numpy as _np

        def collect_eval(trainer, env, n_episodes=5):
            preds = []
            labels = []
            for _ in range(n_episodes):
                obs = env.reset()
                done = False
                obs_list = []
                rewards = []
                while not done:
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=trainer.device)
                    with torch.no_grad():
                        val = trainer.critic(obs_tensor.unsqueeze(0)).cpu().numpy().squeeze(0)
                        mu, _ = trainer.actor(obs_tensor.unsqueeze(0))
                        mu = mu.cpu().numpy().squeeze(0)
                    # policy action
                    act = mu
                    next_obs, r, done, _ = env.step(act)
                    obs_list.append(float(val))
                    rewards.append(float(r))
                    obs = next_obs
                # compute discounted returns per timestep for episode
                returns = []
                G = 0.0
                for rew in reversed(rewards):
                    G = rew + trainer.cfg.gamma * G
                    returns.insert(0, G)
                # pad/truncate to match lengths
                preds.extend(obs_list)
                labels.extend(returns)
            return _np.asarray(preds, dtype=float), _np.asarray(labels, dtype=float)

        preds_arr, labels_arr = collect_eval(trainer, env, n_episodes=5)
        # save alongside checkpoint
        base = os.path.splitext(args.save)[0]
        try:
            _np.save(f"{base}.preds.npy", preds_arr)
            _np.save(f"{base}.val_labels.npy", labels_arr)
        except Exception:
            pass
    except Exception:
        pass

    print("training finished", metrics)


if __name__ == "__main__":
    main()
