"""Unit tests for news sentiment tool."""
from unittest.mock import patch
import pytest


def test_news_sentiment_no_key(monkeypatch):
    monkeypatch.setattr("src.tools.news_sentiment.settings.FINNHUB_API_KEY", "")
    from src.tools.news_sentiment import get_news_sentiment
    result = get_news_sentiment("AAPL")
    assert "articles" in result
    assert result["overall_sentiment"] in {"positive", "negative", "neutral"}


@patch("src.tools.news_sentiment.requests.get")
def test_news_sentiment_with_finnhub(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.news_sentiment.settings.FINNHUB_API_KEY", "test_key")
    mock_get.return_value = type("R", (), {
        "ok": True,
        "raise_for_status": lambda: None,
        "json": lambda: [
            {"headline": "Apple beats earnings", "source": "CNBC", "datetime": 1700000000, "url": "http://example.com"},
            {"headline": "Apple stock surges", "source": "Reuters", "datetime": 1700001000, "url": "http://example.com"},
        ],
    })()

    from src.tools.news_sentiment import get_news_sentiment
    result = get_news_sentiment("AAPL")
    assert len(result["articles"]) == 2
    assert result["overall_sentiment"] == "positive"
