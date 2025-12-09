"""Generate an experiments.txt file for sweeping seeds, action-scales, and obs-modes.

Usage:
  python tooling/generate_experiments.py --out tooling/examples/experiments.txt --seeds 0 1 2 --action-scales 1 5 --obs-modes raw returns

This writes one shell command per line suitable for `tooling/bench_runner.py`.
"""
from __future__ import annotations

import argparse
from pathlib import Path


TEMPLATE = 'python -m src.models.rl.train_ppo --steps {steps} --save models/{name}.pth --seed {seed} --action-scale {scale} --obs-mode {obs} --verbose'


def generate(out: Path, seeds, scales, obs_modes, steps: int = 500):
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for seed in seeds:
        for scale in scales:
            for obs in obs_modes:
                name = f"ppo_s{seed}_scale{scale}_obs{obs}"
                lines.append(TEMPLATE.format(steps=steps, name=name, seed=seed, scale=scale, obs=obs))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} with {len(lines)} commands")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--action-scales", nargs="+", type=float, default=[1.0])
    p.add_argument("--obs-modes", nargs="+", default=["raw"])
    p.add_argument("--steps", type=int, default=500)
    args = p.parse_args()
    generate(Path(args.out), args.seeds, args.action_scales, args.obs_modes, steps=args.steps)


if __name__ == "__main__":
    main()
