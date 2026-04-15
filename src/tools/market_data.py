"""
Market data tool — wraps yfinance.
"""
import yfinance as yf


def get_market_data(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.info
    hist = t.history(period="6mo")

    price_history = [
        {
            "date": str(idx.date()),
            "open": round(row["Open"], 4),
            "high": round(row["High"], 4),
            "low": round(row["Low"], 4),
            "close": round(row["Close"], 4),
            "volume": int(row["Volume"]),
        }
        for idx, row in hist.iterrows()
    ]

    return {
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "market_cap": info.get("marketCap"),
        "volume": info.get("volume"),
        "avg_volume_10d": info.get("averageVolume10days"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "price_history_6m": price_history,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "company_name": info.get("longName", ticker),
        "currency": info.get("currency", "USD"),
    }
