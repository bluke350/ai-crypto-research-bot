"""Lightweight Optuna tuner scaffold.

This module provides a thin wrapper around Optuna to run short tuning jobs.
Optuna is an optional dependency — the code will raise a clear error if it's
not installed. The tuner expects an objective function that accepts a dict of
hyperparameters and returns a scalar metric (higher is better by default).
"""
from __future__ import annotations

from typing import Callable, Dict, Any, Optional


def _ensure_optuna():
    try:
        import optuna
        return optuna
    except Exception as e:
        raise RuntimeError("optuna is required for tuning; install with 'pip install optuna'") from e


def run_optuna_tuner(objective_fn: Callable[[Dict[str, Any]], float], search_space: Dict[str, Any], n_trials: int = 20, seed: int | None = None, storage: Optional[str] = None, pruner: Optional[str] = None):
    """Run a small Optuna study.

    - objective_fn: callable receiving a dict of sampled params, returning scalar metric
    - search_space: dict describing hyperparameter search spaces. Supported keys:
        - ('uniform', low, high)
        - ('loguniform', low, high)
        - ('int', low, high)
        - ('categorical', [vals])

    Returns the Optuna study object.
    """
    optuna = _ensure_optuna()

    def _objective(trial: "optuna.trial.Trial"):
        params = {}
        for name, spec in search_space.items():
            kind = spec[0]
            if kind == 'uniform':
                params[name] = trial.suggest_float(name, spec[1], spec[2])
            elif kind == 'loguniform':
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
            elif kind == 'int':
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif kind == 'categorical':
                params[name] = trial.suggest_categorical(name, spec[1])
            else:
                raise ValueError(f"unsupported search space kind: {kind}")
        # objective_fn returns scalar (higher is better). Optuna minimizes by default so we return negative.
        val = objective_fn(params)
        # ensure float
        return float(val)

    sampler = optuna.samplers.TPESampler(seed=seed) if seed is not None else optuna.samplers.TPESampler()
    # configure pruner
    pruner_obj = None
    if pruner == "median":
        pruner_obj = optuna.pruners.MedianPruner()
    elif pruner == "asha":
        pruner_obj = optuna.pruners.SuccessiveHalvingPruner()
    else:
        pruner_obj = None
    create_study_kwargs = {'direction': 'maximize', 'sampler': sampler}
    if pruner_obj is not None:
        create_study_kwargs['pruner'] = pruner_obj
    if storage is not None:
        create_study_kwargs['storage'] = storage
    study = optuna.create_study(**create_study_kwargs)
    study.optimize(_objective, n_trials=n_trials)
    return study
from typing import Dict, Any, Callable, Sequence, Tuple
import random
from itertools import product


class BayesianTuner:
    """Lightweight placeholder for a Bayesian tuner API.

    This is intentionally simple: it exposes the common interface used by the
    project (suggest/evaluate) but uses random search under the hood. It is
    easy to replace with a proper optimizer (Optuna, skopt, Ax, etc.) later.
    """

    def __init__(self, param_space: Dict[str, Sequence[Any]], n_trials: int = 20, seed: int = 0):
        self.param_space = param_space
        self.n_trials = int(n_trials)
        random.seed(seed)

    def suggest(self) -> Dict[str, Any]:
        """Return a random suggestion sampled from the param_space."""
        return {k: random.choice(list(v)) for k, v in self.param_space.items()}

    def optimize(self, objective: Callable[[Dict[str, Any]], float]) -> Tuple[Dict[str, Any], float]:
        """Run `n_trials` random evaluations of `objective` and return best params and score.

        The objective should take a dict of parameters and return a scalar score to maximize.
        """
        best_score = None
        best_params = None
        # If the param space is small, exhaustively evaluate all combos to be deterministic.
        keys = list(self.param_space.keys())
        values = [list(self.param_space[k]) for k in keys]
        try:
            total = 1
            for v in values:
                total *= max(1, len(v))
        except Exception:
            total = None

        if total is not None and total <= self.n_trials:
            # enumerate all combinations
            for combo in product(*values):
                params = {k: combo[i] for i, k in enumerate(keys)}
                try:
                    score = float(objective(params))
                except Exception:
                    score = float('-inf')
                if best_score is None or score > best_score:
                    best_score = score
                    best_params = params
        else:
            for _ in range(self.n_trials):
                params = self.suggest()
                try:
                    score = float(objective(params))
                except Exception:
                    score = float('-inf')
                if best_score is None or score > best_score:
                    best_score = score
                    best_params = params
        # ensure we always return non-None values (fallback to empty dict/neg-inf)
        if best_params is None:
            best_params = {}
        if best_score is None:
            best_score = float('-inf')
        return best_params, best_score


class RandomSearchTuner(BayesianTuner):
    """Alias for the placeholder tuner to make intent explicit in code."""
    pass


class OptunaTuner:
    """Adapter around Optuna to expose `suggest()` and `optimize(objective)` similar to other tuners.

    Uses the `run_optuna_tuner` helper to avoid duplication; converts param_space
    where necessary into the expected Optuna search_space spec.
    """

    def __init__(self, param_space: Dict[str, Any], n_trials: int = 20, seed: int | None = None, db_url: str | None = None, pruner: str | None = None):
        self.param_space = param_space
        self.n_trials = int(n_trials)
        self.seed = seed
        self.db_url = db_url
        self.pruner = pruner or "median"

    def _to_optuna_search_space(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        search = {}
        for name, spec in param_space.items():
            # If user provided a list/sequence of discrete values, use categorical
            if isinstance(spec, (list, set, tuple)) and not (isinstance(spec, tuple) and isinstance(spec[0], str) and spec[0].lower() in {"uniform", "loguniform", "int", "categorical"}):
                search[name] = ("categorical", list(spec))
            elif isinstance(spec, tuple) and len(spec) >= 1 and isinstance(spec[0], str) and spec[0].lower() in {"uniform", "loguniform", "int", "categorical"}:
                # trust the user-provided spec which matches run_optuna_tuner expectations
                search[name] = spec
            else:
                # fallback: treat as categorical if iterable, else categorical single-value
                try:
                    vals = list(spec)
                    search[name] = ("categorical", vals)
                except Exception:
                    search[name] = ("categorical", [spec])
        return search

    def suggest(self) -> Dict[str, Any]:
        # Suggest via deterministic sampling: pick a random selection from categorical lists.
        import random
        rand = random.Random(self.seed)
        out = {}
        for k, v in self.param_space.items():
            if isinstance(v, (list, set, tuple)) and not (isinstance(v, tuple) and isinstance(v[0], str)):
                out[k] = rand.choice(list(v))
            elif isinstance(v, tuple) and v[0] == "categorical":
                out[k] = rand.choice(list(v[1]))
            else:
                # fallback: take first or use value directly
                if hasattr(v, "__iter__") and not isinstance(v, str):
                    try:
                        out[k] = list(v)[0]
                    except Exception:
                        out[k] = v
                else:
                    out[k] = v
        return out

    def optimize(self, objective) -> tuple:
        # run optuna tuner using helper run_optuna_tuner; convert search_space spec
        try:
            import optuna  # ensure runtime error is meaningful if not installed
        except Exception as exc:
            raise RuntimeError("optuna is required for OptunaTuner; install via 'pip install optuna'") from exc

        search_space = self._to_optuna_search_space(self.param_space)
        # configure pruner
        pruner_obj = None
        if self.pruner == "median":
            pruner_obj = optuna.pruners.MedianPruner()
        elif self.pruner == "asha":
            pruner_obj = optuna.pruners.SuccessiveHalvingPruner()
        else:
            pruner_obj = None
        # storage: allow sqlite URL or None for in-memory
        storage = None
        if self.db_url:
            storage = self.db_url
        study = run_optuna_tuner(objective, search_space, n_trials=self.n_trials, seed=self.seed, storage=storage, pruner=self.pruner)
        best_params = study.best_params if hasattr(study, 'best_params') else {}
        best_score = study.best_value if hasattr(study, 'best_value') else None
        return best_params, float(best_score) if best_score is not None else float('-inf')

