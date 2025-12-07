import math
import pytest

# Skip the test early if optuna isn't available in the environment.
try:
    import optuna  # type: ignore
except Exception:
    pytest.skip("optuna not installed, skipping tuner tests", allow_module_level=True)

from src.tuning.optimizers import run_optuna_tuner


def test_run_optuna_tuner_with_dummy_objective():
    # objective has maximum at x=2 (we maximize). We'll search over a small range.
    def obj(p):
        x = float(p.get('x', 0.0))
        return -(x - 2.0) ** 2

    search = {'x': ('uniform', -5.0, 5.0)}
    study = run_optuna_tuner(obj, search_space=search, n_trials=10, seed=123)
    assert hasattr(study, 'best_trial')
    # best value should be close to 0 (since -(x-2)^2 max = 0 at x=2)
    # allow a small tolerance since the budget is small
    assert math.isclose(study.best_value, 0.0, abs_tol=2e-3)
