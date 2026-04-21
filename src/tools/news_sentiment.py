"""
News sentiment tool — Finnhub company news + sentiment.

Classification priority per article
------------------------------------
1. Finnhub /news-sentiment endpoint  — returns a pre-computed score (0–1)
   for the whole symbol; used as the *overall* score anchor when available.
2. Keyword classifier (_classify_sentiment) — fast, zero-cost, works for
   headlines that contain clear positive/negative vocabulary.
3. LLM batch classifier (_classify_with_llm) — called once for all headlines
   that the keyword classifier returned as neutral (no matching keywords).
   Requires OPENAI_API_KEY; skipped silently if key is absent.

Public API
----------
get_news_sentiment(ticker) -> dict
"""
import time
import json
import requests

from src.config import settings

FINNHUB_BASE = "https://finnhub.io/api/v1"

_POSITIVE = {"beat", "surge", "record", "growth", "profit", "upgrade",
             "strong", "gain", "rally", "soar", "outperform", "exceed",
             "raise", "bullish", "rebound"}
_NEGATIVE = {"miss", "drop", "loss", "decline", "downgrade", "weak",
             "cut", "fail", "slump", "plunge", "underperform", "warning",
             "bearish", "layoff", "recall"}


def _classify_sentiment(headline: str) -> tuple[str, float]:
    """Keyword-based classifier.  Returns (label, score).

    Score magnitude is proportional to keyword count:
      positive → +0.3 + 0.1 * pos_count
      negative → −0.3 − 0.1 * neg_count
      neutral  → 0.0
    """
    words = set(headline.lower().split())
    pos = len(words & _POSITIVE)
    neg = len(words & _NEGATIVE)
    if pos > neg:
        return "positive", round(0.3 + 0.1 * pos, 2)
    if neg > pos:
        return "negative", round(-0.3 - 0.1 * neg, 2)
    return "neutral", 0.0


def _classify_with_llm(headlines: list[str]) -> list[tuple[str, float]]:
    """Batch-classify headlines that the keyword classifier left as neutral.

    Makes a single OpenAI chat completion with all headlines at once.
    Returns a list of (label, score) in the same order as *headlines*.
    Falls back to ("neutral", 0.0) per headline on any error or missing key.
    """
    if not settings.OPENAI_API_KEY or not headlines:
        return [("neutral", 0.0)] * len(headlines)

    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    prompt = (
        "Classify the sentiment of each news headline as positive, negative, or neutral "
        "from an equity investor's perspective. "
        "Reply ONLY with a JSON array of objects, one per headline, in order. "
        'Each object must have "label" (positive/negative/neutral) and '
        '"score" (float: positive→0.1 to 1.0, negative→-1.0 to -0.1, neutral→0.0).\n\n'
        f"Headlines:\n{numbered}"
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        results = []
        for item in parsed:
            label = item.get("label", "neutral").lower()
            score = float(item.get("score", 0.0))
            results.append((label, round(score, 3)))
        if len(results) == len(headlines):
            return results
    except Exception:
        pass

    return [("neutral", 0.0)] * len(headlines)


def _finnhub_overall_sentiment(ticker: str) -> float | None:
    """Fetch Finnhub's pre-computed news sentiment score for *ticker*.

    Endpoint: /news-sentiment
    Returns the ``sentiment.score`` field (0 = very negative, 1 = very positive)
    normalised to our −1 … +1 scale, or None on any error / missing key.
    """
    if not settings.FINNHUB_API_KEY:
        return None
    try:
        resp = requests.get(
            f"{FINNHUB_BASE}/news-sentiment",
            params={"symbol": ticker, "token": settings.FINNHUB_API_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        score = resp.json().get("sentiment", {}).get("score")
        if score is not None:
            # Finnhub score: 0–1 where 0.5 is neutral → map to −1…+1
            return round((float(score) - 0.5) * 2, 3)
    except Exception:
        pass
    return None


def get_news_sentiment(ticker: str) -> dict:
    """Fetch recent news for *ticker* and return sentiment analysis.

    Return schema
    -------------
    {
        "articles": [
            {
                "headline": str,
                "source": str,
                "datetime": str,       # "YYYY-MM-DD HH:MM" UTC
                "url": str,
                "sentiment": str,      # "positive" | "negative" | "neutral"
                "sentiment_score": float,
            },
            ...
        ],
        "overall_sentiment": str,
        "overall_score": float,
        "sentiment_summary": str,
    }
    """
    articles: list[dict] = []

    if settings.FINNHUB_API_KEY:
        end_ts = int(time.time())
        start_ts = end_ts - 7 * 24 * 3600  # last 7 days
        params = {
            "symbol": ticker,
            "from": time.strftime("%Y-%m-%d", time.gmtime(start_ts)),
            "to":   time.strftime("%Y-%m-%d", time.gmtime(end_ts)),
            "token": settings.FINNHUB_API_KEY,
        }
        try:
            resp = requests.get(f"{FINNHUB_BASE}/company-news", params=params, timeout=10)
            resp.raise_for_status()
            news_items = resp.json()[:20]
            for item in news_items:
                sentiment, score = _classify_sentiment(item.get("headline", ""))
                articles.append({
                    "headline":        item.get("headline", ""),
                    "source":          item.get("source", ""),
                    "datetime":        time.strftime(
                                           "%Y-%m-%d %H:%M",
                                           time.gmtime(item.get("datetime", 0))
                                       ),
                    "url":             item.get("url", ""),
                    "sentiment":       sentiment,
                    "sentiment_score": score,
                })
        except Exception:
            pass

    # ── LLM upgrade: reclassify neutral articles ─────────────────────────────
    neutral_idxs = [i for i, a in enumerate(articles) if a["sentiment"] == "neutral"]
    if neutral_idxs:
        neutral_headlines = [articles[i]["headline"] for i in neutral_idxs]
        llm_results = _classify_with_llm(neutral_headlines)
        for idx, (label, score) in zip(neutral_idxs, llm_results):
            articles[idx]["sentiment"]       = label
            articles[idx]["sentiment_score"] = score

    # ── Fallback placeholder ─────────────────────────────────────────────────
    if not articles:
        articles = [{
            "headline":        f"No recent news available for {ticker}",
            "source":          "N/A",
            "datetime":        "",
            "url":             "",
            "sentiment":       "neutral",
            "sentiment_score": 0.0,
        }]

    # ── Overall score: prefer Finnhub's pre-computed score ───────────────────
    finnhub_score = _finnhub_overall_sentiment(ticker)
    article_avg   = round(
        sum(a["sentiment_score"] for a in articles) / len(articles), 3
    )
    overall_score = finnhub_score if finnhub_score is not None else article_avg

    overall = (
        "positive" if overall_score >  0.1 else
        "negative" if overall_score < -0.1 else
        "neutral"
    )

    pos_count = sum(1 for a in articles if a["sentiment"] == "positive")
    neg_count = sum(1 for a in articles if a["sentiment"] == "negative")
    score_src = "Finnhub" if finnhub_score is not None else "article average"
    summary = (
        f"Of {len(articles)} recent articles, {pos_count} were positive "
        f"and {neg_count} were negative. "
        f"Overall sentiment is {overall} ({score_src} score: {overall_score})."
    )

    return {
        "articles":          articles,
        "overall_sentiment": overall,
        "overall_score":     overall_score,
        "sentiment_summary": summary,
    }
