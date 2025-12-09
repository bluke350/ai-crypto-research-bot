import torch
from src.models.transformer_seq import build_transformer, TORCH_AVAILABLE
from src.models.meta_learning import MetaLearner


def test_meta_learner_adapts():
    assert TORCH_AVAILABLE
    model = build_transformer({'input_dim':4, 'd_model':16, 'nhead':4, 'num_layers':1})
    ml = MetaLearner(lr=1e-3, adapt_steps=3)
    seq_len = 8
    batch = 4
    x = torch.randn(seq_len, batch, 4)
    y = torch.randn(seq_len, batch, 1)
    state = ml.adapt(model, x, y)
    assert isinstance(state, dict)
    assert len(state) > 0
