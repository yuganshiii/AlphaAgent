"""Unit tests for market_data tool."""
from unittest.mock import patch, MagicMock
import pytest

from src.tools.market_data import get_market_data


@patch("src.tools.market_data.yf.Ticker")
def test_get_market_data_returns_expected_keys(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "currentPrice": 150.0,
        "marketCap": 2_400_000_000_000,
        "volume": 50_000_000,
        "averageVolume10days": 55_000_000,
        "fiftyTwoWeekHigh": 200.0,
        "fiftyTwoWeekLow": 120.0,
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "longName": "Apple Inc.",
        "currency": "USD",
    }
    import pandas as pd
    mock_ticker.history.return_value = pd.DataFrame()
    mock_ticker_cls.return_value = mock_ticker

    result = get_market_data("AAPL")
    assert result["current_price"] == 150.0
    assert result["sector"] == "Technology"
    assert "price_history_6m" in result
