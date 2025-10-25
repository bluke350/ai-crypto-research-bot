import os
import sys

# Ensure repository root is on sys.path so tests can import `src` package
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_configure(config):
    # Skip real sleeping in RateLimiter during test runs to avoid long delays
    os.environ.setdefault("SKIP_RATE_LIMIT", "1")
    # backoff test mode removed; tests should use run_connect_attempts helper where needed
