#!/usr/bin/env python3
"""Train a tiny PyTorch model and save `models/opportunity.pth` and a metadata JSON.

The metadata contains a minimal spec so `tooling/online_update.py` can reconstruct
the same architecture for warm-start fine-tuning (simple Linear network).
"""
from __future__ import annotations

import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


class SimpleNet(nn.Module):
    def __init__(self, input_dim: int = 3, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def main():
    # synthetic data
    X = np.vstack([
        np.random.normal(loc=-1.0, scale=0.5, size=(200, 3)),
        np.random.normal(loc=1.0, scale=0.5, size=(200, 3)),
    ]).astype('float32')
    y = np.hstack([np.zeros(200), np.ones(200)]).astype('float32')

    model = SimpleNet(input_dim=3, hidden=16)
    criterion = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).unsqueeze(1)

    model.train()
    for epoch in range(50):
        opt.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        opt.step()

    # save state_dict and metadata
    pth = MODEL_DIR / 'opportunity.pth'
    torch.save(model.state_dict(), pth)
    meta = {
        'arch': 'SimpleNet',
        'input_dim': 3,
        'hidden': 16,
        'format': 'state_dict'
    }
    meta_p = MODEL_DIR / 'opportunity.meta.json'
    with open(meta_p, 'w') as f:
        json.dump(meta, f)
    print('Saved PyTorch model and metadata to', pth, meta_p)


if __name__ == '__main__':
    main()
