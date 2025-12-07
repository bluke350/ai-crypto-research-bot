from __future__ import annotations

import json
import os
import time
from typing import Any, Dict


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def emit(event: str, payload: Dict[str, Any], dst: str | None = None) -> None:
    """Emit a simple NDJSON metric/event to `dst` (defaults to `experiments/metrics.log`).

    This is intentionally minimal for PoC. Each line is a JSON object with keys:
    `ts`, `event`, `payload`.
    """
    dst = dst or os.getenv("METRICS_LOG", "experiments/metrics.log")
    _ensure_dir(dst)
    rec = {"ts": int(time.time()), "event": event, "payload": payload}
    try:
        with open(dst, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        # best-effort; emitter must not raise in production flows
        try:
            with open(dst + ".err", "a", encoding="utf-8") as eh:
                eh.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass
