import sys
import types

from src.tuning import optimizers


def test_run_optuna_tuner_with_fake_optuna(monkeypatch):
    # Build a minimal fake optuna module
    fake_optuna = types.SimpleNamespace()

    class FakeSampler:
        def __init__(self, seed=None):
            self.seed = seed

    fake_optuna.samplers = types.SimpleNamespace(TPESampler=FakeSampler)

    class FakeTrial:
        def suggest_float(self, name, low, high, log=False):
            return (low + high) / 2.0

        def suggest_int(self, name, low, high):
            return int(low)

        def suggest_categorical(self, name, values):
            return values[0]

    class FakeStudy:
        def __init__(self):
            self.trials_run = 0

        def optimize(self, objective, n_trials=1):
            # Call the objective n_trials times with a FakeTrial
            for _ in range(n_trials):
                objective(FakeTrial())
                self.trials_run += 1

    def fake_create_study(direction='maximize', sampler=None):
        return FakeStudy()

    fake_optuna.create_study = fake_create_study

    # Inject into sys.modules so _ensure_optuna imports it
    monkeypatch.setitem(sys.modules, 'optuna', fake_optuna)

    # Define a simple search space and objective
    search_space = {'a': ('uniform', 0, 10), 'b': ('int', 1, 3), 'c': ('categorical', ['x', 'y'])}

    def objective(params):
        # a -> 5.0, b -> 1, c -> 'x' (ignored)
        return float(params['a']) + float(params['b'])

    study = optimizers.run_optuna_tuner(objective, search_space, n_trials=5, seed=42)
    assert hasattr(study, 'trials_run')
    assert study.trials_run == 5
