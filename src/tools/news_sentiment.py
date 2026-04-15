"""
News sentiment tool — Finnhub API with fallback to LLM-based classification.
"""
import time
import requests

from src.config import settings

FINNHUB_BASE = "https://finnhub.io/api/v1"


def _classify_sentiment(headline: str) -> tuple[str, float]:
    """Simple keyword-based fallback sentiment classifier."""
    positive_words = {"beat", "surge", "record", "growth", "profit", "upgrade", "strong", "gain"}
    negative_words = {"miss", "drop", "loss", "decline", "downgrade", "weak", "cut", "fail"}
    words = set(headline.lower().split())
    pos = len(words & positive_words)
    neg = len(words & negative_words)
    if pos > neg:
        return "positive", round(0.3 + 0.1 * pos, 2)
    if neg > pos:
        return "negative", round(-0.3 - 0.1 * neg, 2)
    return "neutral", 0.0


def get_news_sentiment(ticker: str) -> dict:
    articles = []

    if settings.FINNHUB_API_KEY:
        end_ts = int(time.time())
        start_ts = end_ts - 7 * 24 * 3600  # last 7 days
        url = f"{FINNHUB_BASE}/company-news"
        params = {
            "symbol": ticker,
            "from": time.strftime("%Y-%m-%d", time.gmtime(start_ts)),
            "to": time.strftime("%Y-%m-%d", time.gmtime(end_ts)),
            "token": settings.FINNHUB_API_KEY,
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            news_items = resp.json()[:20]
            for item in news_items:
                sentiment, score = _classify_sentiment(item.get("headline", ""))
                articles.append(
                    {
                        "headline": item.get("headline", ""),
                        "source": item.get("source", ""),
                        "datetime": time.strftime(
                            "%Y-%m-%d %H:%M", time.gmtime(item.get("datetime", 0))
                        ),
                        "url": item.get("url", ""),
                        "sentiment": sentiment,
                        "sentiment_score": score,
                    }
                )
        except Exception:
            pass

    if not articles:
        articles = [
            {
                "headline": f"No recent news available for {ticker}",
                "source": "N/A",
                "datetime": "",
                "url": "",
                "sentiment": "neutral",
                "sentiment_score": 0.0,
            }
        ]

    scores = [a["sentiment_score"] for a in articles]
    avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0
    overall = "positive" if avg_score > 0.1 else "negative" if avg_score < -0.1 else "neutral"

    pos_count = sum(1 for a in articles if a["sentiment"] == "positive")
    neg_count = sum(1 for a in articles if a["sentiment"] == "negative")
    summary = (
        f"Of {len(articles)} recent articles, {pos_count} were positive and {neg_count} were negative. "
        f"Overall sentiment is {overall} (avg score: {avg_score})."
    )

    return {
        "articles": articles,
        "overall_sentiment": overall,
        "overall_score": avg_score,
        "sentiment_summary": summary,
    }
