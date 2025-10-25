from src.tuning.optimizers import RandomSearchTuner


def test_random_search_tuner_exhaustive_small_grid():
    # small deterministic param space (2 x 2 = 4 combos)
    param_space = {"a": [1, 2], "b": [0.1, 0.2]}
    tuner = RandomSearchTuner(param_space=param_space, n_trials=10, seed=0)


    def objective(params):
        # simple objective: maximize a + b
        return float(params["a"]) + float(params["b"])


    best_params, best_score = tuner.optimize(objective)

    # best should be a=2, b=0.2 => score 2.2
    assert isinstance(best_params, dict)
    assert best_params["a"] == 2
    # float comparison tolerance
    assert abs(best_score - 2.2) < 1e-9
