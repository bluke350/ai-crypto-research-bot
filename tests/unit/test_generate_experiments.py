from pathlib import Path
from tooling.generate_experiments import generate


def test_generate_creates_file(tmp_path):
    out = tmp_path / "examples" / "exp.txt"
    seeds = [0, 1]
    scales = [1.0, 5.0]
    obs = ["raw", "returns"]
    generate(out, seeds, scales, obs, steps=10)
    assert out.exists()
    txt = out.read_text(encoding="utf-8")
    lines = [l for l in txt.splitlines() if l.strip()]
    # combinations = 2*2*2 = 8
    assert len(lines) == 8
