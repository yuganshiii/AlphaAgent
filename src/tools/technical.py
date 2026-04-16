"""
Technical analysis tool — RSI(14), MACD(12,26,9), Bollinger Bands(20,2),
SMA-50/200, EMA-20, ATR(14), Volume SMA(20).

Uses the 'ta' library for all indicator math. Fetches 2 years of daily
OHLCV data by default so SMA-200 has a full warmup period.
Accepts a pre-fetched price_history list to avoid a redundant yfinance call
when market_data has already been pulled.
"""
import math
import pandas as pd
import ta
import yfinance as yf


# ── helpers ───────────────────────────────────────────────────────────────────

def _f(val, digits: int = 4) -> float | None:
    """Float-safe round; returns None for NaN / None / non-numeric."""
    try:
        v = float(val)
        return None if math.isnan(v) or math.isinf(v) else round(v, digits)
    except (TypeError, ValueError):
        return None


def _last(series: pd.Series, digits: int = 4) -> float | None:
    """Return the last non-NaN value of a Series, or None."""
    try:
        clean = series.dropna()
        if clean.empty:
            return None
        return _f(clean.iloc[-1], digits)
    except Exception:
        return None


# ── data loading ──────────────────────────────────────────────────────────────

def _load_df(ticker: str, price_history: list[dict] | None) -> pd.DataFrame:
    """
    Return a OHLCV DataFrame indexed by date.

    If price_history is supplied (from market_data tool's 6-month history)
    we still fetch 2y from yfinance so SMA-200 has enough bars — the
    price_history override is only honoured when it has ≥ 200 rows.
    """
    if price_history and len(price_history) >= 200:
        df = pd.DataFrame(price_history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df.columns = [c.capitalize() for c in df.columns]
        return df

    t = yf.Ticker(ticker)
    df = t.history(period="2y", auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data returned for '{ticker}'.")
    return df


# ── scoring ───────────────────────────────────────────────────────────────────

def _overall_signal(
    price: float,
    sma50: float | None,
    sma200: float | None,
    macd_hist: float | None,
    rsi: float | None,
    bb_upper: float | None,
    bb_lower: float | None,
) -> tuple[str, dict]:
    """
    Score 6 independent signals; return (signal_label, score_breakdown).

    Each signal contributes +1 (bullish), -1 (bearish), or 0 (neutral).
    Total range: [-6, +6].
    Threshold: score > 1 → bullish, score < -1 → bearish, else neutral.
    """
    scores: dict[str, int] = {}

    # 1. SMA-50/200 cross
    if sma50 is not None and sma200 is not None:
        scores["sma_cross"] = 1 if sma50 > sma200 else -1
    else:
        scores["sma_cross"] = 0

    # 2. Price vs SMA-50
    if sma50 is not None:
        scores["price_vs_sma50"] = 1 if price > sma50 else -1
    else:
        scores["price_vs_sma50"] = 0

    # 3. Price vs SMA-200
    if sma200 is not None:
        scores["price_vs_sma200"] = 1 if price > sma200 else -1
    else:
        scores["price_vs_sma200"] = 0

    # 4. MACD histogram direction
    if macd_hist is not None:
        scores["macd_histogram"] = 1 if macd_hist > 0 else -1
    else:
        scores["macd_histogram"] = 0

    # 5. RSI level (contrarian at extremes)
    if rsi is not None:
        if rsi < 30:
            scores["rsi"] = 1      # oversold → bullish
        elif rsi > 70:
            scores["rsi"] = -1     # overbought → bearish
        else:
            scores["rsi"] = 0
    else:
        scores["rsi"] = 0

    # 6. Bollinger Band position
    if bb_upper is not None and bb_lower is not None:
        if price < bb_lower:
            scores["bollinger"] = 1    # below lower band → oversold / mean-revert
        elif price > bb_upper:
            scores["bollinger"] = -1   # above upper band → overbought / mean-revert
        else:
            scores["bollinger"] = 0
    else:
        scores["bollinger"] = 0

    total = sum(scores.values())
    label = "bullish" if total > 1 else "bearish" if total < -1 else "neutral"
    return label, {**scores, "total": total}


# ── main function ─────────────────────────────────────────────────────────────

def get_technical_signals(
    ticker: str,
    price_history: list[dict] | None = None,
) -> dict:
    """
    Compute technical indicators for a ticker.

    Args:
        ticker:        Ticker symbol.
        price_history: Optional pre-fetched OHLCV list from market_data tool.
                       Must have ≥ 200 rows to skip the yfinance fetch.

    Returns:
        {
            "trend": {
                "current_price": float,
                "sma_50": float | None,
                "sma_200": float | None,
                "ema_20": float | None,
                "golden_cross": bool | None,     # SMA-50 > SMA-200
                "death_cross": bool | None,
                "price_vs_sma50": str,           # "above" | "below" | "unknown"
                "price_vs_sma200": str,
                "price_vs_ema20": str,
            },
            "momentum": {
                "rsi_14": float | None,          # 0–100
                "rsi_signal": str,               # "overbought"|"oversold"|"neutral"
                "macd_line": float | None,
                "macd_signal_line": float | None,
                "macd_histogram": float | None,
                "macd_trend": str,               # "bullish"|"bearish"
                "macd_crossover": str,           # "bullish_cross"|"bearish_cross"|"none"
            },
            "volatility": {
                "bb_upper": float | None,
                "bb_middle": float | None,       # SMA-20
                "bb_lower": float | None,
                "bb_width": float | None,        # (upper-lower)/middle — normalised width
                "bb_percent_b": float | None,    # 0=at lower, 1=at upper, <0 or >1 outside
                "atr_14": float | None,
                "atr_pct": float | None,         # ATR as % of current price
                "price_vs_bb": str,              # "above_upper"|"in_band"|"below_lower"
            },
            "volume": {
                "volume_sma_20": float | None,
                "volume_ratio": float | None,    # latest volume / SMA-20
                "volume_trend": str,             # "increasing"|"decreasing"|"stable"
            },
            "overall_signal": str,               # "bullish"|"bearish"|"neutral"
            "signal_scores": dict,               # per-indicator breakdown + total
        }

    Raises:
        ValueError: if no price data is available for the ticker.
    """
    ticker = ticker.strip().upper()
    df = _load_df(ticker, price_history)

    close  = df["Close"].astype(float)
    high   = df["High"].astype(float)
    low    = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    price = _f(close.iloc[-1])
    if price is None:
        raise ValueError(f"Could not read current price for '{ticker}'.")

    # ── trend ─────────────────────────────────────────────────────────────────
    sma50  = _last(ta.trend.sma_indicator(close, window=50))
    sma200 = _last(ta.trend.sma_indicator(close, window=200))
    ema20  = _last(ta.trend.ema_indicator(close, window=20))

    golden_cross = (sma50 > sma200) if (sma50 is not None and sma200 is not None) else None
    death_cross  = (sma50 < sma200) if (sma50 is not None and sma200 is not None) else None

    def _vs(ref: float | None) -> str:
        if ref is None:
            return "unknown"
        return "above" if price > ref else "below"

    trend = {
        "current_price": price,
        "sma_50": sma50,
        "sma_200": sma200,
        "ema_20": ema20,
        "golden_cross": golden_cross,
        "death_cross": death_cross,
        "price_vs_sma50": _vs(sma50),
        "price_vs_sma200": _vs(sma200),
        "price_vs_ema20": _vs(ema20),
    }

    # ── momentum ──────────────────────────────────────────────────────────────
    rsi_series  = ta.momentum.rsi(close, window=14)
    rsi_val     = _last(rsi_series, digits=2)
    rsi_signal  = (
        "overbought" if rsi_val is not None and rsi_val > 70
        else "oversold"  if rsi_val is not None and rsi_val < 30
        else "neutral"
    )

    macd_obj    = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_line   = _last(macd_obj.macd())
    macd_sig    = _last(macd_obj.macd_signal())
    macd_hist   = _last(macd_obj.macd_diff())
    macd_trend  = (
        "bullish" if macd_hist is not None and macd_hist > 0
        else "bearish" if macd_hist is not None and macd_hist < 0
        else "neutral"
    )

    # MACD crossover: compare last two histogram values
    macd_crossover = "none"
    try:
        hist_clean = macd_obj.macd_diff().dropna()
        if len(hist_clean) >= 2:
            prev_hist = _f(hist_clean.iloc[-2])
            if prev_hist is not None and macd_hist is not None:
                if prev_hist < 0 and macd_hist > 0:
                    macd_crossover = "bullish_cross"
                elif prev_hist > 0 and macd_hist < 0:
                    macd_crossover = "bearish_cross"
    except Exception:
        pass

    momentum = {
        "rsi_14": rsi_val,
        "rsi_signal": rsi_signal,
        "macd_line": macd_line,
        "macd_signal_line": macd_sig,
        "macd_histogram": macd_hist,
        "macd_trend": macd_trend,
        "macd_crossover": macd_crossover,
    }

    # ── volatility ────────────────────────────────────────────────────────────
    bb_obj    = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_upper  = _last(bb_obj.bollinger_hband())
    bb_middle = _last(bb_obj.bollinger_mavg())
    bb_lower  = _last(bb_obj.bollinger_lband())

    # Normalised bandwidth: (upper - lower) / middle
    bb_width = (
        _f((bb_upper - bb_lower) / bb_middle)
        if bb_upper is not None and bb_lower is not None and bb_middle
        else None
    )

    # %B: where price sits within the band (0 = lower, 1 = upper)
    bb_percent_b = (
        _f((price - bb_lower) / (bb_upper - bb_lower))
        if bb_upper is not None and bb_lower is not None and (bb_upper - bb_lower) != 0
        else None
    )

    price_vs_bb = "in_band"
    if bb_upper is not None and price > bb_upper:
        price_vs_bb = "above_upper"
    elif bb_lower is not None and price < bb_lower:
        price_vs_bb = "below_lower"

    atr_series = ta.volatility.average_true_range(high, low, close, window=14)
    atr_val    = _last(atr_series)
    atr_pct    = _f(atr_val / price) if atr_val is not None and price else None

    volatility = {
        "bb_upper": bb_upper,
        "bb_middle": bb_middle,
        "bb_lower": bb_lower,
        "bb_width": bb_width,
        "bb_percent_b": bb_percent_b,
        "atr_14": atr_val,
        "atr_pct": atr_pct,
        "price_vs_bb": price_vs_bb,
    }

    # ── volume ────────────────────────────────────────────────────────────────
    vol_sma_series = ta.trend.sma_indicator(volume, window=20)
    vol_sma_val    = _last(vol_sma_series, digits=0)
    latest_vol     = _f(volume.iloc[-1], digits=0)

    vol_ratio = (
        _f(latest_vol / vol_sma_val)
        if latest_vol and vol_sma_val and vol_sma_val > 0
        else None
    )

    # Trend: compare 5-day avg to 20-day SMA
    try:
        recent_avg = float(volume.iloc[-5:].mean())
        vol_trend = (
            "increasing" if vol_sma_val and recent_avg > vol_sma_val * 1.1
            else "decreasing" if vol_sma_val and recent_avg < vol_sma_val * 0.9
            else "stable"
        )
    except Exception:
        vol_trend = "stable"

    vol_section = {
        "volume_sma_20": vol_sma_val,
        "volume_ratio": vol_ratio,
        "volume_trend": vol_trend,
    }

    # ── overall signal ────────────────────────────────────────────────────────
    overall, scores = _overall_signal(
        price=price,
        sma50=sma50,
        sma200=sma200,
        macd_hist=macd_hist,
        rsi=rsi_val,
        bb_upper=bb_upper,
        bb_lower=bb_lower,
    )

    return {
        "trend": trend,
        "momentum": momentum,
        "volatility": volatility,
        "volume": vol_section,
        "overall_signal": overall,
        "signal_scores": scores,
    }
