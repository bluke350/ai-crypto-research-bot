import time
from threading import Lock


class RateLimiter:
    """Simple token-bucket like rate limiter for per-minute quotas.

    Not distributed; intended for single-process orchestration.
    """

    def __init__(self, rate_per_minute: int):
        self.rate = max(1, int(rate_per_minute))
        self.window = 60.0
        self.timestamps = []
        self.lock = Lock()

    def sleep_if_needed(self):
        with self.lock:
            now = time.time()
            self.timestamps = [t for t in self.timestamps if now - t < self.window]
            if len(self.timestamps) >= self.rate:
                sleep_s = self.window - (now - self.timestamps[0])
                time.sleep(max(0.01, sleep_s))
            self.timestamps.append(time.time())
