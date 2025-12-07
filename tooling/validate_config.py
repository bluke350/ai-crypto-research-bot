#!/usr/bin/env python3
"""Validate YAML config(s) against JSON Schema definitions in `src/configs/schema.py`.

Usage:
  python tooling/validate_config.py configs/project.yaml
  python tooling/validate_config.py --schema project configs/project.yaml

Returns exit code 0 on success, non-zero on validation errors.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

try:
    from jsonschema import validate, ValidationError
except Exception:  # pragma: no cover - jsonschema may be absent
    validate = None
    ValidationError = Exception

from src.configs import schema as schema_mod


def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def detect_schema_name(path: Path):
    # heuristic: base name match (project.yaml -> project)
    name = path.stem
    if name in schema_mod.SCHEMAS:
        return name
    # fallback: look for top-level keys
    data = load_yaml(path)
    if not isinstance(data, dict):
        return None
    for key in data.keys():
        if key in schema_mod.SCHEMAS:
            return key
    return None


def run(paths, schema_name=None):
    if validate is None:
        print("jsonschema not installed. Install with: pip install jsonschema")
        return 2

    all_ok = True
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"File not found: {path}")
            all_ok = False
            continue
        data = load_yaml(path)
        s_name = schema_name or detect_schema_name(path)
        if not s_name:
            print(f"No schema found for {path}; skipping")
            continue
        schema = schema_mod.SCHEMAS.get(s_name)
        try:
            validate(instance=data, schema=schema)
            print(f"OK: {path} (schema={s_name})")
        except ValidationError as e:
            print(f"INVALID: {path} (schema={s_name})")
            print(e)
            all_ok = False
    return 0 if all_ok else 3


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="YAML file(s) or glob")
    p.add_argument("--schema", help="Force schema name to validate against")
    args = p.parse_args(argv)
    rc = run(args.paths, schema_name=args.schema)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
