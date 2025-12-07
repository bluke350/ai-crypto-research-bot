import pytest
from src.utils.io import load_prices_csv
import pandas as pd


def test_rl_obs_window_inference_and_generate():
    torch = pytest.importorskip("torch")
    from src.models.rl.policy_adapter import RLPolicyStrategy

    df = load_prices_csv("examples/XBT_USD_prices.csv", dedupe='first')

    strat = RLPolicyStrategy(ckpt_path="models/ppo_smoke.pth")
    assert isinstance(strat.obs_window, int) and strat.obs_window > 0

    targets = strat.generate_targets(df)
    assert len(targets) == len(df)


def test_rl_obs_window_override():
    torch = pytest.importorskip("torch")
    from src.models.rl.policy_adapter import RLPolicyStrategy
    # infer checkpoint input dim and pass it explicitly to ensure override
    data = torch.load("models/ppo_smoke.pth", map_location="cpu")
    actor_sd = data.get("actor", data)
    in_dim = None
    for k, v in actor_sd.items():
        if k.endswith("net.0.weight"):
            in_dim = v.shape[1]
            break
    if in_dim is None:
        for v in actor_sd.values():
            if hasattr(v, "shape") and v.ndim == 2:
                in_dim = v.shape[1]
                break

    strat = RLPolicyStrategy(ckpt_path="models/ppo_smoke.pth", obs_window=int(in_dim))
    assert strat.obs_window == int(in_dim)
