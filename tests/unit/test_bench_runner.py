import csv
import os
import shutil
import tempfile
from pathlib import Path

import tooling.bench_runner as br


def test_run_commands_smoke(tmp_path):
    cmds = [
        "python -c \"print('one')\"",
        "python -c \"print('two')\"",
    ]
    outdir = tmp_path / "bench_out"
    br.run_commands(cmds, str(outdir), parallel=2, timeout=5, retries=0)

    # summary exists
    summary = outdir / "summary.csv"
    assert summary.exists()

    # logs exist and contain the outputs
    logs = list((outdir / "logs").glob("*.log"))
    assert len(logs) == 2
    texts = [p.read_text(encoding='utf-8') for p in logs]
    assert any('one' in t for t in texts)
    assert any('two' in t for t in texts)

    # read summary content
    with open(summary, newline='', encoding='utf-8') as fh:
        r = list(csv.DictReader(fh))
    assert len(r) == 2
    for row in r:
        assert 'cmd' in row and 'log' in row and 'returncode' in row
