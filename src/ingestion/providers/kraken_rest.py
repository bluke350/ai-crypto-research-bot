from typing import List, Dict
import time
import requests
import pandas as pd
from src.utils.rate_limit import RateLimiter
from src.utils.config import load_yaml

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
    while True:
        rl.sleep_if_needed()
        attempts += 1
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data: Dict = resp.json()
        if data.get("error"):
            if attempts >= max_attempts:
                raise RuntimeError(f"Kraken error: {data.get('error')}")
            time.sleep(backoffs[min(attempts - 1, len(backoffs) - 1)] / 1000.0)
            continue
        payload = list(data.get("result", {}).values())[0]
        if not payload:
            break
        for ohlc in payload:
            t, o, h, l, c, v, _, count = ohlc
            if end and t > end:
                break
            rows.append([int(t) * 1000, float(o), float(h), float(l), float(c), float(v), int(count)])
        last_ts = payload[-1][0]
        if end and last_ts >= end:
            break
        params["since"] = last_ts
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "count"])
    if df.empty:
        raise ValueError("Empty OHLC response")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.sort_values("timestamp")
