import builtins
import pytest

from src.tuning import optimizers


def test__ensure_optuna_raises_when_missing(monkeypatch):
    # Simulate ImportError when importing optuna
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'optuna':
            raise ImportError('no optuna')
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', fake_import)

    with pytest.raises(RuntimeError):
        optimizers._ensure_optuna()
