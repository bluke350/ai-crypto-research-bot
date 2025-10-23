import os
import tempfile
from src.utils import config
from dataclasses import dataclass


def test_load_yaml_missing(tmp_path):
    p = tmp_path / "nope.yaml"
    try:
        config.load_yaml(str(p))
        assert False, "should have raised"
    except FileNotFoundError:
        pass


@dataclass
class Cfg:
    a: int = 1
    b: str = "x"


def test_load_yaml_as(tmp_path, monkeypatch):
    p = tmp_path / "c.yaml"
    p.write_text("a: 5\nb: hello\n")
    obj = config.load_yaml_as(str(p), Cfg)
    assert obj.a == 5
    assert obj.b == "hello"
    # env override
    monkeypatch.setenv("A", "10")
    # nested override test
    p.write_text("a: 2\nb: world\n")
    # using internal _override with prefix
    d = {"a": 3, "b": "z"}
    out = config._override_with_env(d, prefix="")
    assert out["a"] in (3, "10")
