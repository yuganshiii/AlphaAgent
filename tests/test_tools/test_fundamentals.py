"""Unit tests for fundamentals tool."""
from unittest.mock import patch, MagicMock
import pytest

from src.tools.fundamentals import get_fundamentals


@patch("src.tools.fundamentals.yf.Ticker")
def test_get_fundamentals_structure(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "totalRevenue": 400_000_000_000,
        "revenueGrowth": 0.08,
        "grossMargins": 0.44,
        "operatingMargins": 0.30,
        "profitMargins": 0.26,
        "trailingEps": 6.42,
        "earningsGrowth": 0.10,
        "totalAssets": 350_000_000_000,
        "totalLiab": 290_000_000_000,
        "totalStockholderEquity": 60_000_000_000,
        "totalCash": 60_000_000_000,
        "totalDebt": 120_000_000_000,
        "trailingPE": 28.5,
        "forwardPE": 26.0,
        "priceToBook": 40.0,
        "priceToSalesTrailing12Months": 7.5,
        "enterpriseToEbitda": 22.0,
        "dividendYield": 0.005,
    }
    mock_ticker_cls.return_value = mock_ticker

    result = get_fundamentals("AAPL")
    assert "income_statement" in result
    assert "balance_sheet" in result
    assert "valuation" in result
    assert result["income_statement"]["revenue_ttm"] == 400_000_000_000
