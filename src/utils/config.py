from __future__ import annotations
import os
from dataclasses import dataclass, fields
from typing import Any, Dict, Type, TypeVar
import yaml

T = TypeVar("T")


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return the parsed object.

    Raises FileNotFoundError if file doesn't exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _override_with_env(cfg: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in cfg.items():
        env_key = (prefix + k).upper()
        if isinstance(v, dict):
            out[k] = _override_with_env(v, prefix=env_key + "_")
        else:
            out[k] = os.getenv(env_key, v)
    return out


def load_dataclass(cls: Type[T], data: Dict[str, Any]) -> T:
    """Instantiate a dataclass from dict with basic type validation.

    Only supports simple fields (int, float, str, bool, dict, list).
    """
    if not hasattr(cls, "__dataclass_fields__"):
        raise TypeError("cls must be a dataclass type")
    init_kwargs = {}
    for f in fields(cls):
        if f.name in data:
            init_kwargs[f.name] = data[f.name]
        elif f.default is not None:
            init_kwargs[f.name] = f.default
        else:
            init_kwargs[f.name] = None
    return cls(**init_kwargs)


def load_yaml_as(path: str, cls: Type[T]) -> T:
    """Load YAML and instantiate into dataclass type with env overrides."""
    raw = load_yaml(path)
    raw = _override_with_env(raw)
    return load_dataclass(cls, raw)
