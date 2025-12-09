import pytest
import pandas as pd
from unittest.mock import patch, Mock

from src.ingestion.providers.kraken_rest import get_ohlc


def make_mock_response(payload):
    mock = Mock()
    mock.raise_for_status = Mock()
    mock.json = Mock(return_value={"error": [], "result": {"XXBTZUSD": payload}})
    return mock


@patch("src.ingestion.providers.kraken_rest.requests.get")
def test_get_ohlc_success(mock_get, tmp_path):
    # small payload: [time, open, high, low, close, vwap, volume, count]
    payload = [[1609459200, "29000", "29100", "28900", "29050", "0", "1.0", 1]]
    mock_get.return_value = make_mock_response(payload)
    df = get_ohlc("XBT/USD", "1m", 1609459200, 1609459300)
    assert not df.empty
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume", "count"]


@patch("src.ingestion.providers.kraken_rest.requests.get")
def test_get_ohlc_empty_raises(mock_get):
    mock = Mock()
    mock.raise_for_status = Mock()
    mock.json = Mock(return_value={"error": [], "result": {"XXBTZUSD": []}})
    mock_get.return_value = mock
    with pytest.raises(ValueError):
        get_ohlc("XBT/USD", "1m", 1609459200, 1609459300)
