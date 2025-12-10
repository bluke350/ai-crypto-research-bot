#!/usr/bin/env python3
"""Pre-commit hook: scan staged Python files (or all files with --all) for naive UTC usage.

Patterns flagged:
- datetime.utcnow(
- pd.Timestamp.utcnow(
- datetime.utcfromtimestamp(
- pd.Timestamp.utcfromtimestamp(

Exit 1 if any matches found.
"""
import re
import sys
import subprocess
from pathlib import Path
import os

# Patterns mapped to friendly suggestions. These catch common naive UTC usages
PATTERNS = [
    (re.compile(r"\bdatetime\.utcnow\s*\("),
     "Use timezone-aware: `src.utils.time.now_utc()` or `datetime.now(timezone.utc)`"),
    (re.compile(r"\bpd\.Timestamp\.utcnow\s*\("),
     "Use `pd.Timestamp.now(tz='UTC')` or `pd.to_datetime(..., utc=True)`"),
    (re.compile(r"\bdatetime\.utcfromtimestamp\s*\("),
     "Use `datetime.fromtimestamp(ts, tz=timezone.utc)`"),
    (re.compile(r"\bpd\.Timestamp\.utcfromtimestamp\s*\("),
     "Use `pd.to_datetime(ts, unit='s', utc=True)` or `pd.Timestamp.fromtimestamp(ts, tz='UTC')`"),
    (re.compile(r"\bdatetime\.now\s*\("),
     "If using `datetime.now()`, pass `tz=timezone.utc` or use `src.utils.time.now_utc()`"),
    (re.compile(r"\bpd\.to_datetime\s*\([^)]*utc\s*=\s*False"),
     "Call `pd.to_datetime(..., utc=True)` to produce timezone-aware timestamps"),
]


def staged_files():
    # return list of staged file paths (relative) that are python files
    out = subprocess.run(["git", "diff", "--cached", "--name-only", "-z"], capture_output=True, text=False)
    if out.returncode != 0:
        return []
    raw = out.stdout
    if not raw:
        return []
    parts = raw.split(b"\0")
    return [p.decode("utf-8") for p in parts if p]


def scan_content(name: str, content: str):
    hits = []
    for i, line in enumerate(content.splitlines(), start=1):
        for pat, suggestion in PATTERNS:
            if pat.search(line):
                hits.append((i, line.strip(), suggestion))
    if hits:
        print(f"{name}:")
        for ln, text, suggestion in hits:
            print(f"  {ln}: {text}")
            print(f"     suggestion: {suggestion}")
    return hits


def main():
    args = sys.argv[1:]
    files = []
    if "--all" in args:
        # scan all tracked python files
        out = subprocess.run(["git", "ls-files", "*.py"], capture_output=True, text=True)
        files = [p.strip() for p in out.stdout.splitlines() if p.strip()]
    else:
        files = [f for f in staged_files() if f.endswith('.py')]

    # skip checking the checker script itself and pre-commit config (it may contain examples)
    files = [f for f in files if os.path.normpath(f) not in (os.path.normpath('scripts/check_naive_utc.py'), '.pre-commit-config.yaml')]

    if not files:
        return 0

    any_hits = False
    for f in files:
        try:
            # read staged content if present, else worktree file
            content_proc = subprocess.run(["git", "show", f":" + f], capture_output=True, text=True)
            content = content_proc.stdout if content_proc.returncode == 0 else Path(f).read_text(encoding='utf-8')
        except Exception:
            try:
                content = Path(f).read_text(encoding='utf-8')
            except Exception:
                continue
        hits = scan_content(f, content)
        if hits:
            any_hits = True

    if any_hits:
        print("\nRejecting commit: found naive UTC usage. Use timezone-aware helpers (e.g. src.utils.time.now_utc()).")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
