"""Example script to train a small TransformerSeq on synthetic data and save the model."""
from __future__ import annotations

import argparse

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--save', type=str, default='models/transformer_toy.pth')
    p.add_argument('--steps', type=int, default=20)
    args = p.parse_args()

    from src.models.transformer_seq import build_transformer, TORCH_AVAILABLE
    if not TORCH_AVAILABLE:
        raise RuntimeError('torch not available; install torch to run this example')
    import torch

    cfg = {'input_dim': 6, 'd_model': 32, 'nhead': 4, 'num_layers': 1}
    model = build_transformer(cfg)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    seq_len = 16
    batch = 8
    input_dim = cfg['input_dim']
    X = torch.randn(seq_len, batch, input_dim)
    Y = torch.randn(seq_len, batch, 1)

    for _ in range(min(args.steps, 100)):
        optimizer.zero_grad()
        out = model.forward(X)
        loss = loss_fn(out, Y)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), args.save)
    print(f"Saved toy transformer to {args.save}")
