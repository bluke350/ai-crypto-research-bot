#!/usr/bin/env python3
"""Fetch latest 'opportunity-model' artifact from GitHub Actions using `gh` CLI.

This script uses the `gh` CLI (GitHub CLI) and the repo remote to locate
the latest run of the `retrain-opportunity.yml` workflow and downloads the
artifact named `opportunity-model` into `models/opportunity.pkl`.

It is a best-effort tool: it only runs if `gh` is installed and the user is
authenticated (via `gh auth login`).
"""
from __future__ import annotations

import json
import subprocess
import sys
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; env may be provided from environment
    pass

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "opportunity.pkl"

HAS_BOTO3 = True
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except Exception:
    HAS_BOTO3 = False


def get_repo_remote() -> str:
    try:
        out = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True, check=True)
        url = out.stdout.strip()
        # parse formats: git@github.com:owner/repo.git or https://github.com/owner/repo.git
        if url.startswith("git@"):
            # git@github.com:owner/repo.git
            _, path = url.split(":", 1)
            owner_repo = path.replace('.git', '')
        else:
            # https://github.com/owner/repo.git
            owner_repo = url.split('github.com/')[-1].replace('.git', '')
        return owner_repo
    except Exception:
        return ""


def gh_available() -> bool:
    try:
        subprocess.run(["gh", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def main() -> int:
    if MODEL_PATH.exists():
        print(f"Model already exists at {MODEL_PATH}")
        return 0

    # Prefer S3 download if S3_BUCKET is configured
    s3_bucket = os.environ.get("S3_BUCKET")
    s3_key = os.environ.get("S3_KEY", "models/opportunity.pkl")
    if s3_bucket:
        if not HAS_BOTO3:
            print("boto3 not installed; cannot fetch from S3", file=sys.stderr)
        else:
            print(f"Attempting to download from s3://{s3_bucket}/{s3_key}")
            try:
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                s3 = boto3.client(
                    's3',
                    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                    region_name=os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION'),
                )
                s3.download_file(s3_bucket, s3_key, str(MODEL_PATH))
                if MODEL_PATH.exists():
                    print(f"Downloaded model to {MODEL_PATH}")
                    return 0
            except (BotoCoreError, ClientError) as exc:
                print(f"S3 download failed: {exc}", file=sys.stderr)
                # Provide helpful debug information for diagnosing 403 / permissions issues
                try:
                    sts = boto3.client('sts')
                    whoami = sts.get_caller_identity()
                    print("AWS caller identity:", json.dumps(whoami))
                except Exception as e2:
                    print("Could not get caller identity:", e2, file=sys.stderr)
                try:
                    loc = s3.get_bucket_location(Bucket=s3_bucket)
                    print("Bucket location:", loc)
                except Exception as e3:
                    print("Could not get bucket location:", e3, file=sys.stderr)

    # Fallback to GitHub Actions artifact download via gh CLI
    if not gh_available():
        print("Neither S3 nor gh are available; cannot fetch artifact", file=sys.stderr)
        return 2

    repo = get_repo_remote()
    if not repo:
        print("Could not determine repo from git remote; aborting", file=sys.stderr)
        return 3

    # Find the latest workflow run for retrain-opportunity.yml
    try:
        # Get latest run id for the workflow file
        runs = subprocess.run(["gh", "api", f"repos/{repo}/actions/workflows/retrain-opportunity.yml/runs", "--jq", ".workflow_runs[0].id"], capture_output=True, text=True, check=True)
        run_id = runs.stdout.strip()
        if not run_id:
            print("No workflow runs found for retrain-opportunity.yml", file=sys.stderr)
            return 4
        print(f"Found workflow run id: {run_id}")

        # Download artifact named 'opportunity-model' from that run
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cmd = ["gh", "run", "download", run_id, "--name", "opportunity-model", "--dir", str(MODEL_DIR)]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        # The artifact is a zip; gh run download should extract into the dir
        if MODEL_PATH.exists():
            print(f"Downloaded model to {MODEL_PATH}")
            return 0
        else:
            # try to find any artifact file under models/
            for p in MODEL_DIR.glob("**/*"):
                if p.is_file() and p.suffix in ('.pkl', '.bin'):
                    p.rename(MODEL_PATH)
                    print(f"Moved {p} -> {MODEL_PATH}")
                    return 0
            print("Artifact downloaded but model file not found in artifact contents", file=sys.stderr)
            return 5
    except subprocess.CalledProcessError as exc:
        print("gh command failed:", exc, file=sys.stderr)
        return 6


if __name__ == '__main__':
    raise SystemExit(main())
