Artifact storage and retrieval

This project supports publishing retrained model artifacts to an S3-compatible storage bucket from GitHub Actions, and fetching them at runtime locally using `tooling/fetch_opportunity_artifact.py`.

Design
- GitHub Actions will always upload a GitHub Actions artifact as a fallback.
- If AWS credentials and `S3_BUCKET` are provided to the workflow, the retrain job will copy the model to `s3://$S3_BUCKET/$S3_KEY`.
- At runtime, `tooling/fetch_opportunity_artifact.py` will attempt to download from S3 when `S3_BUCKET` is configured. If boto3 or S3 credentials aren't available it will fall back to downloading via the `gh` CLI (existing behavior).

Recommended GitHub Secrets
- `AWS_ACCESS_KEY_ID` — IAM key with PutObject permission for the target bucket
- `AWS_SECRET_ACCESS_KEY` — IAM secret
- `AWS_REGION` — AWS region for the bucket (e.g. `us-east-1`)
- `S3_BUCKET` — target bucket name (e.g. `my-repo-artifacts`)
- `S3_KEY` (optional) — key path to store object, defaults to `models/opportunity.pkl`

How to configure
1. Create or identify an S3 bucket and a user/role with permission to write objects to the bucket.
2. Add the secrets above to your GitHub repository (Settings → Secrets and variables → Actions → New repository secret).
3. The existing workflow `.github/workflows/retrain-opportunity.yml` will now attempt to copy the retrained model to the configured S3 bucket.

Local dev (.env)
- For local testing, place an `.env` file at the repository root (the fetcher uses `python-dotenv` if present). Example `.env`:

```
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET=my-repo-artifacts
S3_KEY=models/opportunity.pkl
```

Usage
- Fetch the model into `models/opportunity.pkl`:

```bash
python tooling/fetch_opportunity_artifact.py
```

Notes
- The script prefers S3 download (boto3). If boto3 is not installed it will still try the `gh` CLI approach.
- Do not commit secrets to the repository. Use GitHub Secrets or a secure credential manager for CI.

Git hooks to prevent accidental commits
-------------------------------------
- This repo includes a lightweight pre-commit script at `.githooks/pre-commit` that scans staged files for common AWS credential patterns (e.g., `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AKIA...`).
- To enable the hook locally for this repository run:

```bash
git config core.hooksPath .githooks
```

- After enabling hooks, attempts to commit files that contain likely credentials will be blocked. This is a convenience safeguard — keep rotating credentials and storing secrets in GitHub Secrets.

Versioning and 'latest' pointer
--------------------------------
- The retrain workflow now uploads the artifact to a timestamped key in S3 (e.g. `models/opportunity-20251207T0300Z.pkl`) and also copies the current model to `models/opportunity-latest.pkl`. This keeps a historical record while providing a stable key for runtime fetchers.

Rotation reminder
-----------------
- If you accidentally exposed an access key (even briefly), rotate it immediately in the AWS Console (IAM -> Users -> Security credentials -> Create access key / Delete old key). Update your GitHub Secrets and local `.env` accordingly.

If you want support for other storage providers (GCS, Azure Blob), I can add that as well.
