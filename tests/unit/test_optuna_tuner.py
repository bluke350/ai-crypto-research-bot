import pytest
import numpy as np
import pandas as pd

from src.tuning.optimizers import OptunaTuner


def test_optuna_tuner_smoke():
    pytest.importorskip('optuna')

    # toy objective: prefers x near 0.5
    def objective(params: dict) -> float:
        x = float(params.get('x', 0.5))
        return -(x - 0.5) ** 2

    param_space = {'x': ('uniform', 0.0, 1.0)}
    t = OptunaTuner(param_space, n_trials=10, seed=0)
    best_params, best_score = t.optimize(objective)
    assert 'x' in best_params
    assert isinstance(best_score, float)
