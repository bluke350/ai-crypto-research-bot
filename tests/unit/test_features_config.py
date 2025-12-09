from __future__ import annotations

import os
from pathlib import Path

from src.features.config import load_features_config


def test_load_default_features_config(tmp_path, monkeypatch):
    # ensure default config path resolves to the repo configs file
    repo_configs = Path.cwd() / 'configs' / 'features.yaml'
    assert repo_configs.exists(), 'expected configs/features.yaml to exist in repo'

    cfg = load_features_config()
    assert isinstance(cfg, dict)
    assert 'features' in cfg
    features = cfg['features']
    assert isinstance(features, dict)
    # check some expected top-level keys
    assert 'microstructure' in features
    assert 'volume_profile' in features


def test_load_explicit_path(tmp_path):
    f = tmp_path / 'my_features.yaml'
    f.write_text('features:\n  microstructure:\n    enabled: false\n')
    cfg = load_features_config(str(f))
    assert cfg['features']['microstructure']['enabled'] is False
