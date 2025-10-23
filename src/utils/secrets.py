import os
import yaml
from typing import Dict, Any


def load_credentials(path: str = "configs/credentials.yaml") -> Dict[str, Any]:
    """Load credentials from an untracked YAML file or environment variables.

    Priority: explicit path -> env VARS (CREDENTIALS__NESTED) -> example file absent.
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    # fallback: read env variables with __ separator
    creds = {}
    for k, v in os.environ.items():
        if k.startswith("CREDENTIALS__"):
            parts = k[len("CREDENTIALS__") :].lower().split("__")
            d = creds
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = v
    return creds
