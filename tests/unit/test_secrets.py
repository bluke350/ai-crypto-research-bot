import os
import tempfile
from src.utils import secrets


def test_load_credentials_env(monkeypatch):
    monkeypatch.setenv("CREDENTIALS__KRAKEN__KEY", "abc")
    monkeypatch.setenv("CREDENTIALS__KRAKEN__SECRET", "def")
    d = secrets.load_credentials(path="nonexistent.yaml")
    assert d.get("kraken", {}).get("key") == "abc"
    assert d.get("kraken", {}).get("secret") == "def"
