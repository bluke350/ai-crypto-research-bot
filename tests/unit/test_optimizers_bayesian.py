from src.tuning.optimizers import BayesianTuner


def test_optimize_exhaustive_enumeration():
    # small param space 2x2 = 4 combos, n_trials large enough to trigger exhaustive path
    param_space = {"x": [1, 2], "y": [10, 20]}
    tuner = BayesianTuner(param_space=param_space, n_trials=10, seed=0)

    def objective(params):
        # score is x * 100 + y, unique best is x=2,y=20 => 220
        return params["x"] * 100 + params["y"]

    best_params, best_score = tuner.optimize(objective)
    assert best_params == {"x": 2, "y": 20}
    assert best_score == 220


def test_optimize_handles_objective_exceptions():
    # small param space: objective raises for one combo
    param_space = {"a": [0, 1], "b": [0, 1]}
    tuner = BayesianTuner(param_space=param_space, n_trials=10, seed=0)

    def objective(params):
        # raise when both zero, otherwise return sum
        if params["a"] == 0 and params["b"] == 0:
            raise RuntimeError("bad")
        return params["a"] + params["b"]

    best_params, best_score = tuner.optimize(objective)
    # best should be a=1,b=1 => score 2
    assert best_score == 2
    assert best_params == {"a": 1, "b": 1}


def test_optimize_random_path_returns_values():
    # make param space large so random path is taken
    param_space = {"i": list(range(100)), "j": list(range(100, 200))}
    tuner = BayesianTuner(param_space=param_space, n_trials=5, seed=42)

    def objective(params):
        return float(params["i"]) - float(params["j"]) / 100.0

    best_params, best_score = tuner.optimize(objective)
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)
