from __future__ import annotations

from typing import Dict, Any, Optional
try:
    import torch
    TORCH = True
except Exception:
    torch = None
    TORCH = False


class MetaLearner:
    """A tiny meta-learning adapter: fine-tune model's last linear layer on a small support set.

    This is intentionally simple: it performs a small number of SGD steps on the provided
    model and returns the adapted state_dict. It requires PyTorch.
    """

    def __init__(self, lr: float = 1e-3, adapt_steps: int = 5):
        if not TORCH:
            raise RuntimeError('torch is required for MetaLearner')
        self.lr = float(lr)
        self.adapt_steps = int(adapt_steps)

    def adapt(self, model: 'torch.nn.Module', support_x: 'torch.Tensor', support_y: 'torch.Tensor') -> Dict[str, Any]:
        """Fine-tune `model` on (support_x, support_y) and return adapted state_dict.

        support_x expected shape: (seq_len, batch, input_dim) or (batch, seq_len, input_dim)
        support_y expected shape: matching model output.
        """
        if not TORCH:
            raise RuntimeError('torch is required for MetaLearner')
        model = model
        optim = torch.optim.SGD(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        model.train()
        for _ in range(self.adapt_steps):
            optim.zero_grad()
            out = model.forward(support_x)
            loss = loss_fn(out, support_y)
            loss.backward()
            optim.step()
        return {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}


__all__ = ['MetaLearner']
