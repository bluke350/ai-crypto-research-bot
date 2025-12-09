"""Transformer sequence model scaffold.

This file provides a PyTorch-based TransformerSeq class when `torch` is available.
If `torch` is not installed, the module defines a placeholder that raises a clear error on use.

The scaffold is intentionally small and designed so importing this module does not fail
if `torch` is missing (useful for lightweight unit tests in CI without heavy deps).
"""
from __future__ import annotations

from typing import Optional, Dict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False


class TransformerSeq(nn.Module):
    """Simple Transformer encoder-decoder scaffold for sequence prediction.

    Usage:
        if not TORCH_AVAILABLE:
            raise RuntimeError('torch is required to instantiate TransformerSeq')
        model = TransformerSeq(input_dim=8, d_model=64, nhead=4, num_layers=2)
    """

    def __init__(self, input_dim: int = 8, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        if not TORCH_AVAILABLE:
            raise RuntimeError('torch not available: install torch to use TransformerSeq')
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """Expect `x` of shape (seq_len, batch, input_dim). Returns (seq_len, batch, 1) predictions."""
        x = self.input_proj(x)
        # positional encoding could be added here
        out = self.encoder(x)
        return self.output_proj(out)

    def predict(self, x_numpy, device: Optional[str] = 'cpu'):
        """Convenience wrapper that accepts numpy arrays and returns numpy preds."""
        if not TORCH_AVAILABLE:
            raise RuntimeError('torch not available: install torch to use TransformerSeq')
        import numpy as _np
        xt = torch.tensor(x_numpy, dtype=torch.float32).to(device)
        if xt.dim() == 3:
            # assume shape (batch, seq_len, input_dim) -> transpose to (seq_len, batch, input_dim)
            xt = xt.permute(1, 0, 2)
        with torch.no_grad():
            out = self.forward(xt)
        # out shape (seq_len, batch, 1) -> return last time step of first batch
        out_np = out.cpu().numpy()
        return out_np


def build_transformer(config: Optional[Dict] = None) -> TransformerSeq:
    cfg = config or {}
    input_dim = int(cfg.get('input_dim', 8))
    d_model = int(cfg.get('d_model', 64))
    nhead = int(cfg.get('nhead', 4))
    num_layers = int(cfg.get('num_layers', 2))
    dropout = float(cfg.get('dropout', 0.1))
    return TransformerSeq(input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)


__all__ = ['TransformerSeq', 'build_transformer', 'TORCH_AVAILABLE']
