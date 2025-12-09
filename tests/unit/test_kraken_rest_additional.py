import time
from unittest.mock import Mock

import pandas as pd

from src.ingestion.providers import kraken_rest as kr


def _make_resp(json_value):
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json = Mock(return_value=json_value)
    return resp


def test_get_ohlc_empty_response_raises(monkeypatch):
    # ensure rate limiter and sleeps are no-ops in tests
    monkeypatch.setattr(kr.rl, "sleep_if_needed", lambda: None)
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    # response with empty payload (result contains pair -> empty list)
    data = {"error": [], "result": {"XXBTZUSD": []}}
    monkeypatch.setattr(kr, "requests", Mock(get=Mock(return_value=_make_resp(data))))

    try:
        kr.get_ohlc("XBT/USD", "1m", since=1)
        raise AssertionError("Expected ValueError for empty OHLC response")
    except ValueError as e:
        assert "Empty OHLC response" in str(e)


def test_get_ohlc_retry_then_success(monkeypatch):
    # first response contains an error -> triggers retry path, second returns payload
    monkeypatch.setattr(kr.rl, "sleep_if_needed", lambda: None)
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    err = {"error": ["Temporary failure"], "result": {}}
    ok = {
        "error": [],
        "result": {"XXBTZUSD": [[1600000000, "100", "110", "90", "105", "0", "10.0", 1]]},
    }

    call_seq = [ _make_resp(err), _make_resp(ok) ]

    def _get(url, params=None, timeout=None):
        return call_seq.pop(0)

    monkeypatch.setattr(kr, "requests", Mock(get=_get))

    df = kr.get_ohlc("XBT/USD", "1m", since=1600000000)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    # timestamp column is datetime UTC
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]) or pd.api.types.is_datetime64tz_dtype(df["timestamp"])


def test_get_trades_empty_response_raises(monkeypatch):
    monkeypatch.setattr(kr.rl, "sleep_if_needed", lambda: None)
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    data = {"error": [], "result": {"XXBTZUSD": []}}
    monkeypatch.setattr(kr, "requests", Mock(get=Mock(return_value=_make_resp(data))))

    try:
        kr.get_trades("XBT/USD", since=1)
        raise AssertionError("Expected ValueError for empty trades response")
    except ValueError as e:
        assert "Empty trades response" in str(e)


def test_get_trades_handles_non_numeric_last(monkeypatch):
    # when 'last' is non-numeric, ensure function handles it gracefully
    monkeypatch.setattr(kr.rl, "sleep_if_needed", lambda: None)
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    payload = [["100.0", "0.5", 1600000000.0, "b", "", ""]]
    data = {"error": [], "result": {"XXBTZUSD": payload, "last": "LASTTOKEN"}}
    # return the payload once, then return an empty payload so the loop exits
    call_seq = [_make_resp(data), _make_resp({"error": [], "result": {"XXBTZUSD": []}})]

    def _get(url, params=None, timeout=None):
        return call_seq.pop(0) if call_seq else _make_resp({"error": [], "result": {"XXBTZUSD": []}})

    monkeypatch.setattr(kr, "requests", Mock(get=_get))

    df = kr.get_trades("XBT/USD", since=1600000000)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "price" in df.columns and "size" in df.columns
    # timestamp normalized to datetime
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]) or pd.api.types.is_datetime64tz_dtype(df["timestamp"])
