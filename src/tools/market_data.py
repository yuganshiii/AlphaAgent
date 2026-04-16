"""
Market data tool — wraps yfinance.

Returns structured market data for a given ticker. Raises ValueError
for invalid/unknown tickers so the agent can log and degrade gracefully.
"""
import yfinance as yf


def get_market_data(ticker: str) -> dict:
    """
    Fetch current market data for a ticker symbol.

    Returns:
        {
            "ticker": str,
            "company_name": str,
            "current_price": float | None,
            "previous_close": float | None,
            "price_change": float | None,       # absolute $ change vs prev close
            "price_change_pct": float | None,   # % change vs prev close
            "market_cap": int | None,
            "volume": int | None,
            "avg_volume_10d": int | None,
            "volume_ratio": float | None,       # today's volume / 10d avg
            "52w_high": float | None,
            "52w_low": float | None,
            "52w_high_pct": float | None,       # % below 52w high
            "52w_low_pct": float | None,        # % above 52w low
            "beta": float | None,
            "sector": str | None,
            "industry": str | None,
            "currency": str,
            "exchange": str | None,
            "price_history_6m": list[dict],     # [{date, open, high, low, close, volume}]
        }

    Raises:
        ValueError: if the ticker is invalid or yfinance returns no data.
    """
    ticker = ticker.strip().upper()

    t = yf.Ticker(ticker)

    # yfinance doesn't raise on bad tickers — detect via empty info
    try:
        info = t.info
    except Exception as exc:
        raise ValueError(f"Failed to fetch data for '{ticker}': {exc}") from exc

    # A valid ticker always has at least one of these populated
    current_price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("navPrice")       # for ETFs/funds
    )
    if current_price is None and not info.get("longName"):
        raise ValueError(
            f"Ticker '{ticker}' not found or no market data available. "
            "Check the symbol and try again."
        )

    # 6-month price history
    try:
        hist = t.history(period="6mo", auto_adjust=True)
    except Exception:
        hist = None

    price_history = []
    if hist is not None and not hist.empty:
        for idx, row in hist.iterrows():
            price_history.append({
                "date": str(idx.date()),
                "open": _round(row.get("Open")),
                "high": _round(row.get("High")),
                "low": _round(row.get("Low")),
                "close": _round(row.get("Close")),
                "volume": _to_int(row.get("Volume")),
            })

    previous_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
    price_change = (
        round(current_price - previous_close, 4)
        if current_price is not None and previous_close
        else None
    )
    price_change_pct = (
        round((price_change / previous_close) * 100, 4)
        if price_change is not None and previous_close
        else None
    )

    volume = _to_int(info.get("volume") or info.get("regularMarketVolume"))
    avg_vol_10d = _to_int(info.get("averageVolume10days") or info.get("averageDailyVolume10Day"))
    volume_ratio = (
        round(volume / avg_vol_10d, 3)
        if volume and avg_vol_10d
        else None
    )

    w52_high = info.get("fiftyTwoWeekHigh")
    w52_low = info.get("fiftyTwoWeekLow")
    w52_high_pct = (
        round(((w52_high - current_price) / w52_high) * 100, 2)
        if w52_high and current_price
        else None
    )
    w52_low_pct = (
        round(((current_price - w52_low) / w52_low) * 100, 2)
        if w52_low and current_price
        else None
    )

    return {
        "ticker": ticker,
        "company_name": info.get("longName") or info.get("shortName") or ticker,
        "current_price": _round(current_price),
        "previous_close": _round(previous_close),
        "price_change": price_change,
        "price_change_pct": price_change_pct,
        "market_cap": _to_int(info.get("marketCap")),
        "volume": volume,
        "avg_volume_10d": avg_vol_10d,
        "volume_ratio": volume_ratio,
        "52w_high": _round(w52_high),
        "52w_low": _round(w52_low),
        "52w_high_pct": w52_high_pct,
        "52w_low_pct": w52_low_pct,
        "beta": info.get("beta"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "currency": info.get("currency") or "USD",
        "exchange": info.get("exchange") or info.get("fullExchangeName"),
        "price_history_6m": price_history,
    }


# ── helpers ──────────────────────────────────────────────────────────────────

def _round(val, digits: int = 4):
    try:
        return round(float(val), digits)
    except (TypeError, ValueError):
        return None


def _to_int(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return None
