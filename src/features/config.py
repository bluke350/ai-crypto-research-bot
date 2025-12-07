from __future__ import annotations

import os
from typing import Any, Dict

import yaml


def load_features_config(path: str | None = None) -> Dict[str, Any]:
    """Load features configuration YAML.

    Looks for (in order):
    - explicit `path` argument
    - environment variable `FEATURES_CONFIG`
    - repository `configs/features.yaml`

    Returns a dict with the parsed YAML (empty dict on missing file).
    """
    if path is None:
        path = os.environ.get('FEATURES_CONFIG')
    if path is None:
        # default to repo-relative configs/features.yaml
        repo_root = os.getcwd()
        path = os.path.join(repo_root, 'configs', 'features.yaml')

    if not os.path.exists(path):
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    # basic validation/coercion
    if 'features' not in data:
        data = {'features': data}

    return data


__all__ = ['load_features_config']
