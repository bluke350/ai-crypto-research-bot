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

PATTERNS = [
    re.compile(r"\bdatetime\.utcnow\s*\("),
    re.compile(r"\bpd\.Timestamp\.utcnow\s*\("),
    re.compile(r"\bdatetime\.utcfromtimestamp\s*\("),
    re.compile(r"\bpd\.Timestamp\.utcfromtimestamp\s*\("),
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
        for pat in PATTERNS:
            if pat.search(line):
                hits.append((i, line.strip()))
    if hits:
        print(f"{name}:")
        for ln, text in hits:
            print(f"  {ln}: {text}")
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
