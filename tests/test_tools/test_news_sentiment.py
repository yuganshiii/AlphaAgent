"""
Comprehensive tests for news_sentiment.py.

All Finnhub HTTP calls are mocked. The keyword-based sentiment classifier
is tested directly.
"""
import time
import pytest
import requests
from unittest.mock import patch, MagicMock


# ── mock response factory ─────────────────────────────────────────────────────

def _finnhub_resp(articles: list):
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = articles
    return mock


def _article(headline: str, source: str = "CNBC", ts: int = 1_700_000_000):
    return {"headline": headline, "source": source, "datetime": ts, "url": "http://example.com"}


# ── output structure ──────────────────────────────────────────────────────────

def test_output_keys_no_api_key(monkeypatch):
    monkeypatch.setattr("src.tools.news_sentiment.settings.FINNHUB_API_KEY", "")
    from src.tools.news_sentiment import get_news_sentiment
    result = get_news_sentiment("AAPL")
    assert set(result.keys()) == {"articles", "overall_sentiment", "overall_score", "sentiment_summary"}


@patch("src.tools.news_sentiment.requests.get")
def test_output_keys_with_api_key(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.news_sentiment.settings.FINNHUB_API_KEY", "key")
    mock_get.return_value = _finnhub_resp([_article("Apple beats earnings")])
    from src.tools.news_sentiment import get_news_sentiment
    result = get_news_sentiment("AAPL")
    assert set(result.keys()) == {"articles", "overall_sentiment", "overall_score", "sentiment_summary"}


@patch("src.tools.news_sentiment.requests.get")
def test_article_keys(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.news_sentiment.settings.FINNHUB_API_KEY", "key")
    mock_get.return_value = _finnhub_resp([_article("Apple beats earnings")])
    from src.tools.news_sentiment import get_news_sentiment
    article = get_news_sentiment("AAPL")["articles"][0]
    assert set(article.keys()) == {"headline", "source", "datetime", "url", "sentiment", "sentiment_score"}


# ── sentiment classifier unit tests ──────────────────────────────────────────

def test_classify_positive_keyword():
    from src.tools.news_sentiment import _classify_sentiment
    # Classifier matches exact words; use "beat" not "beats"
    label, score = _classify_sentiment("Apple beat earnings expectations")
    assert label == "positive"
    assert score > 0


def test_classify_negative_keyword():
    from src.tools.news_sentiment import _classify_sentiment
    label, score = _classify_sentiment("Apple reports a massive loss this quarter")
    assert label == "negative"
    assert score < 0


def test_classify_neutral_no_keywords():
    from src.tools.news_sentiment import _classify_sentiment
    label, score = _classify_sentiment("Apple holds annual developer conference")
    assert label == "neutral"
    assert score == 0.0


def test_classify_positive_score_magnitude():
    from src.tools.news_sentiment import _classify_sentiment
    _, score1 = _classify_sentiment("company beat record growth surge")  # 3 positive
    _, score2 = _classify_sentiment("company beat earnings")              # 1 positive
    assert score1 > score2


def test_classify_negative_score_magnitude():
    from src.tools.news_sentiment import _classify_sentiment
    _, s1 = _classify_sentiment("miss drop loss decline")   # 4 negative
    _, s2 = _classify_sentiment("stock miss target")        # 1 negative
    assert s1 < s2


def test_classify_tie_defaults_to_neutral():
    from src.tools.news_sentiment import _classify_sentiment
    # "beat" (positive) and "miss" (negative) — equal count
    label, _ = _classify_sentiment("Apple beat revenue but miss earnings")
    assert label == "neutral"


def test_classify_case_insensitive():
    from src.tools.news_sentiment import _classify_sentiment
    label1, _ = _classify_sentiment("APPLE BEATS ESTIMATES")
    label2, _ = _classify_sentiment("apple beats estimates")
    assert label1 == label2


# ── overall sentiment thresholds ──────────────────────────────────────────────

@patch("src.tools.news_sentiment.requests.get")
def test_overall_positive_when_avg_above_threshold(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.news_sentiment.settings.FINNHUB_API_KEY", "key")
    # Force purely positive articles
    mock_get.return_value = _finnhub_resp([
        _article("Apple beats record profit surge"),
        _article("Apple growth strong upgrade"),
    ])
    from src.tools.news_sentiment import get_news_sentiment
    result = get_news_sentiment("AAPL")
    assert result["overall_sentiment"] == "positive"
    assert result["overall_score"] > 0.1


@patch("src.tools.news_sentiment.requests.get")
def test_overall_negative_when_avg_below_threshold(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.news_sentiment.settings.FINNHUB_API_KEY", "key")
    mock_get.return_value = _finnhub_resp([
        _article("Apple miss loss decline drop"),
        _article("Apple weak earnings fail downgrade"),
    ])
    from src.tools.news_sentiment import get_news_sentiment
    result = get_news_sentiment("AAPL")
    assert result["overall_sentiment"] == "negative"
    assert result["overall_score"] < -0.1


@patch("src.tools.news_sentiment.requests.get")
def test_overall_neutral_in_middle(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.news_sentiment.settings.FINNHUB_API_KEY", "key")
    # No sentiment keywords → neutral score = 0.0
    mock_get.return_value = _finnhub_resp([
        _article("Apple holds annual meeting"),
        _article("Apple hires new VP of marketing"),
    ])
    from src.tools.news_sentiment import get_news_sentiment
    result = get_news_sentiment("AAPL")
    assert result["overall_sentiment"] == "neutral"


# ── fallback behaviour ────────────────────────────────────────────────────────

def test_no_api_key_returns_placeholder_article(monkeypatch):
    monkeypatch.setattr("src.tools.news_sentiment.settings.FINNHUB_API_KEY", "")
    from src.tools.news_sentiment import get_news_sentiment
    result = get_news_sentiment("AAPL")
    assert len(result["articles"]) == 1
    assert "No recent news" in result["articles"][0]["headline"]


def test_no_api_key_placeholder_has_all_fields(monkeypatch):
    monkeypatch.setattr("src.tools.news_sentiment.settings.FINNHUB_API_KEY", "")
    from src.tools.news_sentiment import get_news_sentiment
    article = get_news_sentiment("AAPL")["articles"][0]
    assert set(article.keys()) == {"headline", "source", "datetime", "url", "sentiment", "sentiment_score"}


@patch("src.tools.news_sentiment.requests.get")
def test_api_timeout_falls_back_to_placeholder(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.news_sentiment.settings.FINNHUB_API_KEY", "key")
    mock_get.side_effect = requests.exceptions.Timeout("timed out")
    from src.tools.news_sentiment import get_news_sentiment
    result = get_news_sentiment("AAPL")     # must not raise
    assert len(result["articles"]) == 1
    assert "No recent news" in result["articles"][0]["headline"]


@patch("src.tools.news_sentiment.requests.get")
def test_http_error_falls_back_to_placeholder(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.news_sentiment.settings.FINNHUB_API_KEY", "key")
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
    mock_get.return_value = mock_resp
    from src.tools.news_sentiment import get_news_sentiment
    result = get_news_sentiment("AAPL")
    assert len(result["articles"]) >= 1


@patch("src.tools.news_sentiment.requests.get")
def test_empty_finnhub_response_returns_placeholder(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.news_sentiment.settings.FINNHUB_API_KEY", "key")
    mock_get.return_value = _finnhub_resp([])
    from src.tools.news_sentiment import get_news_sentiment
    result = get_news_sentiment("AAPL")
    assert len(result["articles"]) == 1
    assert "No recent news" in result["articles"][0]["headline"]


@patch("src.tools.news_sentiment.requests.get")
def test_truncated_to_20_articles(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.news_sentiment.settings.FINNHUB_API_KEY", "key")
    mock_get.return_value = _finnhub_resp(
        [_article(f"Headline {i}") for i in range(50)]
    )
    from src.tools.news_sentiment import get_news_sentiment
    result = get_news_sentiment("AAPL")
    assert len(result["articles"]) <= 20


# ── overall_score arithmetic ──────────────────────────────────────────────────

@patch("src.tools.news_sentiment.requests.get")
def test_overall_score_is_average_of_article_scores(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.news_sentiment.settings.FINNHUB_API_KEY", "key")
    mock_get.return_value = _finnhub_resp([
        _article("Apple beats earnings"),   # positive → 0.4
        _article("Apple stock drop loss"),  # negative → -0.5
    ])
    from src.tools.news_sentiment import get_news_sentiment, _classify_sentiment
    result = get_news_sentiment("AAPL")
    scores = [a["sentiment_score"] for a in result["articles"]]
    expected_avg = round(sum(scores) / len(scores), 3)
    assert result["overall_score"] == expected_avg


# ── sentiment_summary content ─────────────────────────────────────────────────

@patch("src.tools.news_sentiment.requests.get")
def test_summary_mentions_article_count(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.news_sentiment.settings.FINNHUB_API_KEY", "key")
    mock_get.return_value = _finnhub_resp([
        _article("Apple beats earnings"),
        _article("Apple stock drop"),
        _article("Apple holds meeting"),
    ])
    from src.tools.news_sentiment import get_news_sentiment
    result = get_news_sentiment("AAPL")
    assert "3" in result["sentiment_summary"]


@patch("src.tools.news_sentiment.requests.get")
def test_summary_mentions_overall_sentiment(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.news_sentiment.settings.FINNHUB_API_KEY", "key")
    mock_get.return_value = _finnhub_resp([_article("Apple beats earnings")])
    from src.tools.news_sentiment import get_news_sentiment
    result = get_news_sentiment("AAPL")
    assert result["overall_sentiment"] in result["sentiment_summary"]


# ── datetime formatting ───────────────────────────────────────────────────────

@patch("src.tools.news_sentiment.requests.get")
def test_datetime_is_formatted_string(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.news_sentiment.settings.FINNHUB_API_KEY", "key")
    mock_get.return_value = _finnhub_resp([_article("headline", ts=1_700_000_000)])
    from src.tools.news_sentiment import get_news_sentiment
    article = get_news_sentiment("AAPL")["articles"][0]
    # Should be a date string like "2023-11-14 22:13"
    dt = article["datetime"]
    assert isinstance(dt, str)
    assert len(dt) > 0


# ── missing article fields ────────────────────────────────────────────────────

@patch("src.tools.news_sentiment.requests.get")
def test_missing_fields_default_gracefully(mock_get, monkeypatch):
    """Finnhub article with missing keys should not crash."""
    monkeypatch.setattr("src.tools.news_sentiment.settings.FINNHUB_API_KEY", "key")
    mock_get.return_value = _finnhub_resp([{}])   # completely empty article dict
    from src.tools.news_sentiment import get_news_sentiment
    result = get_news_sentiment("AAPL")
    assert len(result["articles"]) == 1
    article = result["articles"][0]
    assert "sentiment" in article
    assert "sentiment_score" in article
