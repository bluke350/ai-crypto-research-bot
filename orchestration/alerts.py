from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import requests

LOG = logging.getLogger(__name__)


def send_webhook(url: str, payload: Dict[str, Any], timeout: int = 5) -> bool:
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return True
    except Exception as e:
        LOG.warning("webhook send failed: %s", e)
        return False


def write_local_alert(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def notify(summary: Dict[str, Any], *, webhook_url: Optional[str] = None, fallback_path: Optional[str] = None) -> bool:
    payload = {
        "type": "run_summary",
        "summary": summary,
    }
    if webhook_url:
        ok = send_webhook(webhook_url, payload)
        if ok:
            return True
    # fallback to file-based alert
    if fallback_path:
        try:
            write_local_alert(fallback_path, payload)
            return True
        except Exception as e:
            LOG.warning("fallback write failed: %s", e)
    return False
