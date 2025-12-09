from typing import List, Dict
import time
import pandas as pd

from src.utils.rate_limit import RateLimiter
from src.utils.config import load_yaml


# provide a lightweight fallback when 'requests' is not installed so tests
# can still import this module and patch requests.get
try:
    import requests
except Exception:
    class _DummyRequests:
        class RequestException(Exception):
            pass

        @staticmethod
        def get(*args, **kwargs):
            raise _DummyRequests.RequestException("requests not installed in test environment")

    requests = _DummyRequests()


# load rate limits from config
cfg = load_yaml("configs/kraken.yaml").get("kraken", {})
rl = RateLimiter(rate_per_minute=cfg.get("rate_limits", {}).get("rest_per_minute", 60))


def _interval_to_minutes(s: str) -> int:
    return {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}.get(s, 1)


def get_ohlc(pair: str, interval: str, since: int, end: int = None) -> pd.DataFrame:
    """Fetch OHLC from Kraken REST API. Returns DataFrame with timestamp(ms), open, high, low, close, volume, count.

    This is a synchronous blocking call. Retries and backoff are basic. Tests should mock requests.get.
    """
    base = cfg.get("base_url", "https://api.kraken.com")
    url = f"{base}/0/public/OHLC"
    params = {"pair": pair, "interval": _interval_to_minutes(interval), "since": since}
    rows: List[List] = []
    attempts = 0
    max_attempts = cfg.get("retries", {}).get("max_attempts", 5)
    backoffs = cfg.get("retries", {}).get("backoff_ms", [200, 400, 800])
    import random
    while True:
        rl.sleep_if_needed()
        attempts += 1
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data: Dict = resp.json()
        if data.get("error"):
            if attempts >= max_attempts:
                raise RuntimeError(f"Kraken error: {data.get('error')}")
            # add jitter to backoff to avoid thundering herd
            base_ms = backoffs[min(attempts - 1, len(backoffs) - 1)]
            jitter = random.uniform(0.5, 1.5)
            time.sleep((base_ms * jitter) / 1000.0)
            continue
        payload = list(data.get("result", {}).values())[0]
        if not payload:
            break
        for ohlc in payload:
            # Kraken returns [time, open, high, low, close, vwap, volume, count]
            t = ohlc[0]
            if end and t > end:
                break
            # be defensive with types
            rows.append([int(t) * 1000, float(ohlc[1]), float(ohlc[2]), float(ohlc[3]), float(ohlc[4]), float(ohlc[6]), int(ohlc[7] if len(ohlc) > 7 else 0)])
        last_ts = payload[-1][0]
        # safety: if server returns the same last_ts as our 'since' param repeatedly,
        # break to avoid infinite loops (useful for tests that stub requests.get).
        if end and last_ts >= end:
            break
        prev_since = params.get("since")
        # If the server didn't advance beyond our previous 'since', stop to avoid a tight loop
        if prev_since is not None and int(prev_since) == int(last_ts):
            break
        params["since"] = last_ts
        # small random sleep to avoid deterministic tight loops
        time.sleep(0.2 + random.uniform(0.0, 0.1))
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "count"])
    if df.empty:
        raise ValueError("Empty OHLC response")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.sort_values("timestamp")


def get_trades(pair: str, since: int, end: int = None) -> pd.DataFrame:
    """Fetch recent trades (trade-level) from Kraken REST public trades endpoint.

    Returns DataFrame with columns: timestamp (s), price, size, side (if present)
    """
    base = cfg.get("base_url", "https://api.kraken.com")
    url = f"{base}/0/public/Trades"
    params = {"pair": pair, "since": since}
    rows = []
    attempts = 0
    max_attempts = cfg.get("retries", {}).get("max_attempts", 5)
    backoffs = cfg.get("retries", {}).get("backoff_ms", [200, 400, 800])
    while True:
        rl.sleep_if_needed()
        attempts += 1
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            if attempts >= max_attempts:
                raise RuntimeError(f"Kraken error: {data.get('error')}")
            time.sleep(backoffs[min(attempts - 1, len(backoffs) - 1)] / 1000.0)
            continue
        # response includes 'result': { '<pair>': [ [price, volume, time, side, ordertype, misc], ... ], 'last': '...'}
        result = data.get("result", {})
        payload = None
        for k, v in result.items():
            if k == "last":
                continue
            payload = v
            break
        if not payload:
            break
        for t in payload:
            # t: [price, volume, time, side, ordertype, misc]
            price = float(t[0])
            size = float(t[1])
            ts = float(t[2])
            side = t[3] if len(t) > 3 else None
            # ts is seconds with fractional part; keep as float seconds
            rows.append([ts, price, size, side])
        last = result.get("last")
        if not last:
            break
        # Kraken 'since' uses a token; some APIs expect the 'last' token
        params["since"] = last
        if end:
            # if 'end' provided (epoch seconds), stop once last >= end
            try:
                if float(last) >= float(end):
                    break
            except Exception:
                pass
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=["timestamp", "price", "size", "side"]) if rows else pd.DataFrame(columns=["timestamp", "price", "size", "side"])
    if df.empty:
        raise ValueError("Empty trades response")
    # normalize timestamp to seconds (float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df.sort_values("timestamp")
