from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        # include additional attributes attached to record (conservative)
        for k in ("ctx", "extra", "meta"):
            v = getattr(record, k, None)
            if v is not None and isinstance(v, dict):
                payload.update(v)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_structured_logger(name: str = __name__, *, file_path: Optional[str] = None, level: int = logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # avoid duplicate handlers
    if logger.handlers:
        return logger

    fmt = JsonFormatter()
    stream_h = logging.StreamHandler()
    stream_h.setFormatter(fmt)
    logger.addHandler(stream_h)

    if file_path:
        file_h = RotatingFileHandler(file_path, maxBytes=5 * 1024 * 1024, backupCount=3)
        file_h.setFormatter(fmt)
        logger.addHandler(file_h)

    return logger
