import numpy as np
import torch
from src.models.transformer_seq import build_transformer, TORCH_AVAILABLE


def test_transformer_forward_pass():
    assert TORCH_AVAILABLE
    cfg = {'input_dim': 4, 'd_model': 16, 'nhead': 4, 'num_layers': 1}
    model = build_transformer(cfg)
    # create synthetic input: shape (seq_len, batch, input_dim)
    seq_len = 12
    batch = 2
    x = torch.randn(seq_len, batch, cfg['input_dim'])
    out = model.forward(x)
    # expected shape (seq_len, batch, 1)
    assert out.shape == (seq_len, batch, 1)
