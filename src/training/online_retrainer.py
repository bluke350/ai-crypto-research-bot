"""Minimal online/adaptive retrainer scaffold.

This module provides a lightweight OnlineRetrainer class that monitors metrics
and triggers a retrain callback when configured thresholds are crossed.

The implementation is intentionally simple and synchronous so it can be used
in scripts, CI, or integrated into the `continuous_controller` later.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from collections import deque
import json
import os

from src.persistence.db import RunLogger


@dataclass
class RetrainPolicy:
    """Defines when to trigger retraining.

    - metric: name of the metric to monitor (e.g. 'val_loss', 'sharpe')
    - direction: 'down' if lower is worse (e.g. val_loss), 'up' if higher is worse (e.g. latency)
    - threshold: amount (absolute or relative) to consider as degradation
    - window: number of recent metric points to consider
    - cooldown_seconds: minimum seconds between retrain triggers
    """
    metric: str
    direction: str = 'down'
    threshold: float = 0.01
    window: int = 5
    cooldown_seconds: int = 3600
    # detection_mode: 'mean' (compare to mean of previous window) or 'ewma'
    detection_mode: str = 'mean'
    # EWMA smoothing factor (alpha) used when detection_mode == 'ewma'
    ewma_alpha: float = 0.3


class OnlineRetrainer:
    def __init__(self, retrain_fn: Callable[[Dict], str], policy: RetrainPolicy, run_logger_db: Optional[str] = None):
        """Create an OnlineRetrainer.

        - retrain_fn: callable that performs retraining; receives a dict `context` and must
          return a path to the produced artifact (e.g. model file) or an identifier string.
        - policy: a `RetrainPolicy` instance.
        - run_logger_db: optional SQLAlchemy DB URL to pass to `RunLogger`.
        """
        self.retrain_fn = retrain_fn
        self.policy = policy
        self.db_url = run_logger_db
        self._values = deque(maxlen=policy.window)
        self._last_trigger = 0.0

    def ingest(self, metrics: Dict[str, float]):
        """Ingest a metrics dict (from a batch, monitoring job, or file).

        Returns True if retrain was triggered.
        """
        val = metrics.get(self.policy.metric)
        if val is None:
            return False

        self._values.append(float(val))
        if len(self._values) < 2:
            return False

        if time.time() - self._last_trigger < self.policy.cooldown_seconds:
            # still in cooldown
            return False

        if self._check_trigger():
            # run retrain and record artifact
            artifact = self.retrain_fn({'metrics': list(self._values), 'policy': self.policy.__dict__})
            # log to RunLogger
            run_id = f"auto_retrain_{int(time.time())}"
            with RunLogger(run_id=run_id, cfg={'policy': self.policy.__dict__}, db_url=self.db_url) as rl:
                try:
                    # artifact may be a path
                    if artifact:
                        rl.log_artifact(str(artifact), kind='model')
                    rl.log_metrics({f'retrain_trigger_{self.policy.metric}': self._values[-1]})
                except Exception:
                    # swallow logging errors to avoid crashing monitoring
                    pass
            self._last_trigger = time.time()
            # clear recent values to avoid immediate re-triggers
            self._values.clear()
            return True

        return False

    def ingest_nonblocking(self, metrics: Dict[str, float]):
        """Like `ingest` but does NOT call `retrain_fn` inline.

        Returns a tuple `(triggered: bool, retrain_callable: Optional[Callable[[], str]])`.
        If `triggered` is True, `retrain_callable` is a no-arg callable that when
        invoked will execute the retrain and return an artifact path (or identifier).
        The caller is responsible for executing the callable (optionally in a worker)
        and for logging artifacts/metrics if desired.
        """
        val = metrics.get(self.policy.metric)
        if val is None:
            return False, None

        self._values.append(float(val))
        if len(self._values) < 2:
            return False, None

        if time.time() - self._last_trigger < self.policy.cooldown_seconds:
            return False, None

        if self._check_trigger():
            # prepare retrain callable (capture current metric window and policy)
            context = {'metrics': list(self._values), 'policy': self.policy.__dict__}
            def _callable():
                return self.retrain_fn(context)

            self._last_trigger = time.time()
            self._values.clear()
            return True, _callable

        return False, None

    def _check_trigger(self) -> bool:
        """Detection logic supporting both 'mean' and 'ewma' modes.

        - 'mean': compare latest to mean of previous window (default behavior).
        - 'ewma': compute EWMA over the previous values (excluding latest) and
          compare latest to EWMA baseline.
        """
        vals = list(self._values)
        if len(vals) < 2:
            return False
        latest = vals[-1]

        if self.policy.detection_mode == 'ewma':
            # compute EWMA over prior values (exclude latest)
            prior = vals[:-1]
            if not prior:
                return False
            alpha = float(self.policy.ewma_alpha)
            ewma = prior[0]
            for v in prior[1:]:
                ewma = alpha * v + (1 - alpha) * ewma
            baseline = ewma
        else:
            prior = vals[:-1]
            baseline = sum(prior) / max(1, len(prior))

        if self.policy.direction == 'down':
            # worse if latest greater than baseline by threshold
            return latest > baseline * (1.0 + self.policy.threshold)
        else:
            return latest < baseline * (1.0 - self.policy.threshold)


def example_retrain_fn(context: Dict) -> str:
    """Example retrain function used by the CLI for testing.

    It writes a small artifact file and returns its path.
    """
    from pathlib import Path

    out_dir = Path('experiments/trained_models')
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = out_dir / f'auto_model_{ts}.txt'
    with open(path, 'w', encoding='utf-8') as f:
        f.write('retrained at %d\n' % ts)
        f.write(json.dumps(context))
    return str(path)
