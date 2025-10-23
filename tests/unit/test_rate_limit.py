import time
from src.utils.rate_limit import RateLimiter


def test_rate_limiter_basic():
    rl = RateLimiter(rate_per_minute=5)
    start = time.time()
    for _ in range(5):
        rl.sleep_if_needed()
    # immediate calls within rate should be fast
    assert time.time() - start < 1.0
