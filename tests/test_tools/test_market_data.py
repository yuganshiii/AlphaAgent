"""Unit tests for market_data tool."""
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from src.tools.market_data import get_market_data


def _make_ticker(info: dict, history: pd.DataFrame = None):
    mock = MagicMock()
    mock.info = info
    mock.history.return_value = history if history is not None else pd.DataFrame()
    return mock


VALID_INFO = {
    "longName": "Apple Inc.",
    "currentPrice": 195.50,
    "previousClose": 193.00,
    "marketCap": 3_000_000_000_000,
    "volume": 60_000_000,
    "averageVolume10days": 55_000_000,
    "fiftyTwoWeekHigh": 220.00,
    "fiftyTwoWeekLow": 160.00,
    "beta": 1.25,
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "currency": "USD",
    "exchange": "NASDAQ",
}


@patch("src.tools.market_data.yf.Ticker")
def test_returns_all_expected_keys(mock_cls):
    mock_cls.return_value = _make_ticker(VALID_INFO)
    result = get_market_data("AAPL")

    expected_keys = {
        "ticker", "company_name", "current_price", "previous_close",
        "price_change", "price_change_pct", "market_cap", "volume",
        "avg_volume_10d", "volume_ratio", "52w_high", "52w_low",
        "52w_high_pct", "52w_low_pct", "beta", "sector", "industry",
        "currency", "exchange", "price_history_6m",
    }
    assert expected_keys == set(result.keys())


@patch("src.tools.market_data.yf.Ticker")
def test_price_and_change_calculated_correctly(mock_cls):
    mock_cls.return_value = _make_ticker(VALID_INFO)
    result = get_market_data("AAPL")

    assert result["current_price"] == 195.50
    assert result["previous_close"] == 193.00
    assert result["price_change"] == round(195.50 - 193.00, 4)
    assert result["price_change_pct"] == round((2.50 / 193.00) * 100, 4)


@patch("src.tools.market_data.yf.Ticker")
def test_52w_percentages_calculated(mock_cls):
    mock_cls.return_value = _make_ticker(VALID_INFO)
    result = get_market_data("AAPL")

    # % below 52w high
    assert result["52w_high_pct"] == round(((220.00 - 195.50) / 220.00) * 100, 2)
    # % above 52w low
    assert result["52w_low_pct"] == round(((195.50 - 160.00) / 160.00) * 100, 2)


@patch("src.tools.market_data.yf.Ticker")
def test_volume_ratio_calculated(mock_cls):
    mock_cls.return_value = _make_ticker(VALID_INFO)
    result = get_market_data("AAPL")
    assert result["volume_ratio"] == round(60_000_000 / 55_000_000, 3)


@patch("src.tools.market_data.yf.Ticker")
def test_price_history_parsed(mock_cls):
    import numpy as np
    dates = pd.date_range("2024-10-01", periods=3, freq="B")
    hist = pd.DataFrame({
        "Open": [190.0, 191.0, 192.0],
        "High": [195.0, 196.0, 197.0],
        "Low":  [189.0, 190.0, 191.0],
        "Close":[193.0, 194.0, 195.0],
        "Volume": [50_000_000, 52_000_000, 48_000_000],
    }, index=dates)

    mock_cls.return_value = _make_ticker(VALID_INFO, hist)
    result = get_market_data("AAPL")

    assert len(result["price_history_6m"]) == 3
    first = result["price_history_6m"][0]
    assert set(first.keys()) == {"date", "open", "high", "low", "close", "volume"}
    assert first["close"] == 193.0
    assert isinstance(first["volume"], int)


@patch("src.tools.market_data.yf.Ticker")
def test_ticker_is_uppercased(mock_cls):
    mock_cls.return_value = _make_ticker(VALID_INFO)
    result = get_market_data("aapl")
    assert result["ticker"] == "AAPL"


@patch("src.tools.market_data.yf.Ticker")
def test_invalid_ticker_raises_value_error(mock_cls):
    # yfinance returns empty info for unknown tickers
    mock_cls.return_value = _make_ticker({})
    with pytest.raises(ValueError, match="not found"):
        get_market_data("INVALIDXYZ")


@patch("src.tools.market_data.yf.Ticker")
def test_info_fetch_exception_raises_value_error(mock_cls):
    mock = MagicMock()
    mock.info = property(lambda self: (_ for _ in ()).throw(Exception("network error")))
    type(mock).info = property(lambda self: (_ for _ in ()).throw(Exception("network error")))
    mock_cls.return_value = mock
    with pytest.raises(ValueError):
        get_market_data("AAPL")


@patch("src.tools.market_data.yf.Ticker")
def test_missing_optional_fields_return_none(mock_cls):
    # Minimal info — only enough to pass validation
    mock_cls.return_value = _make_ticker({
        "longName": "Some Corp",
        "currentPrice": 50.0,
    })
    result = get_market_data("XYZ")
    assert result["beta"] is None
    assert result["sector"] is None
    assert result["volume_ratio"] is None
    assert result["price_change"] is None
    assert result["52w_high_pct"] is None


@patch("src.tools.market_data.yf.Ticker")
def test_fallback_to_regular_market_price(mock_cls):
    info = {**VALID_INFO}
    del info["currentPrice"]
    info["regularMarketPrice"] = 188.00
    mock_cls.return_value = _make_ticker(info)
    result = get_market_data("AAPL")
    assert result["current_price"] == 188.00
