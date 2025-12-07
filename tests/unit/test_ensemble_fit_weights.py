import numpy as np
from src.models.ensemble_manager import EnsembleManager


def test_fit_weights_from_preds_prefers_lower_mse():
    names = ['m1', 'm2']
    em = EnsembleManager(names)
    # create synthetic preds: m1 is perfect, m2 is noisy
    y = np.array([1.0, 2.0, 3.0, 4.0])
    preds = {
        'm1': np.array([1.0, 2.0, 3.0, 4.0]),
        'm2': np.array([1.1, 1.9, 3.2, 3.7]),
    }
    weights = em.fit_weights_from_preds(preds, y, metric='mse')
    # m1 should have higher weight than m2
    assert weights['m1'] > weights['m2']
    # weights should sum to ~1
    assert abs(sum(weights.values()) - 1.0) < 1e-6
