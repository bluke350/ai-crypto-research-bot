"""JSON Schema definitions for configs/*.yaml and a lightweight validator.

This module exposes a `SCHEMAS` mapping suitable for validating common
project/data/kraken config files. Expand the schemas as your config surface
grows.
"""
from __future__ import annotations

SCHEMAS = {
    "project": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "project": {
                "type": "object",
                "properties": {
                    "experiment_db": {"type": "string"},
                    "artifacts_dir": {"type": "string"},
                },
                "required": ["experiment_db"],
            }
        },
        "required": ["project"],
    },
    "data": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "bar_interval": {"type": "string"},
                    "provider": {"type": "string"},
                },
                "required": ["bar_interval"],
            }
        },
        "required": ["data"],
    },
    "kraken": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "kraken": {
                "type": "object",
                "properties": {
                    "symbols": {"type": "array", "items": {"type": "string"}},
                    "bar_interval": {"type": "string"},
                    "ws_url": {"type": "string"},
                },
                "required": ["symbols"],
            }
        },
        "required": ["kraken"],
    },
}
