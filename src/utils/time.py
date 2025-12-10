from datetime import datetime, timezone


def now_utc() -> datetime:
    """Return current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def now_iso() -> str:
    """Return current time as an ISO-8601 string in UTC."""
    return now_utc().isoformat()


def from_timestamp(ts: float) -> datetime:
    """Convert epoch seconds to a timezone-aware UTC datetime."""
    return datetime.fromtimestamp(ts, tz=timezone.utc)
