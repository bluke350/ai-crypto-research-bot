import time
import tempfile
from pathlib import Path

import pytest

from src.training.online_retrainer import OnlineRetrainer, RetrainPolicy


def test_ewma_triggers_retrain(tmp_path):
    calls = []

    def fake_retrain_fn(context):
        # record call and write a tiny artifact
        calls.append(context)
        p = tmp_path / f"artifact_{len(calls)}.txt"
        p.write_text('ok')
        return str(p)

    policy = RetrainPolicy(metric='val_loss', direction='down', threshold=0.05, window=5, cooldown_seconds=3600, detection_mode='ewma', ewma_alpha=0.5)
    retrainer = OnlineRetrainer(retrain_fn=fake_retrain_fn, policy=policy, run_logger_db=f"sqlite:///{tmp_path / 'registry.db'}")

    # feed metrics that slowly increase. EWMA should rise and trigger when latest exceeds baseline by threshold
    metrics_seq = [0.10, 0.11, 0.115, 0.12, 0.16]
    for m in metrics_seq:
        retrainer.ingest({'val_loss': m})
    # retrain should have been called once during the sequence
    assert len(calls) == 1


def test_ewma_does_not_trigger_when_within_threshold(tmp_path):
    calls = []

    def fake_retrain_fn(context):
        calls.append(context)
        # return a dummy artifact path (satisfy typed return)
        return str(tmp_path / 'noop.txt')

    policy = RetrainPolicy(metric='val_loss', direction='down', threshold=0.5, window=5, cooldown_seconds=0, detection_mode='ewma', ewma_alpha=0.3)
    retrainer = OnlineRetrainer(retrain_fn=fake_retrain_fn, policy=policy)

    metrics_seq = [0.10, 0.102, 0.101, 0.103, 0.104]
    for m in metrics_seq:
        assert retrainer.ingest({'val_loss': m}) is False
    assert len(calls) == 0
