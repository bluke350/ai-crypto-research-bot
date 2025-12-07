"""Run walk-forward evaluation with the placeholder BayesianTuner.

This script demonstrates wiring the `BayesianTuner` into the walk-forward
function to tune per-fold parameters. It is intentionally small and meant as
an example/harness for integration tests.
"""
from __future__ import annotations

import argparse
from src.tuning.optimizers import BayesianTuner
from src.validation.walk_forward import evaluate_walk_forward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # minimal param space for example
    param_space = {"window": [50, 100], "fast": [5, 10], "slow": [20, 50]}
    tuner = BayesianTuner(param_space, n_trials=10, seed=args.seed)

    # In practice you would load price history and pass a simulator/strategy.
    # Here we simply print that tuner is ready as an integration stub.
    print("BayesianTuner ready; example usage: pass `tuner` and `param_space` into evaluate_walk_forward")


if __name__ == "__main__":
    main()
