"""
Technical analysis tool — RSI, MACD, Bollinger Bands, moving averages.
"""
import pandas as pd
import ta
import yfinance as yf


def get_technical_signals(ticker: str, price_history: list[dict] = None) -> dict:
    if price_history:
        df = pd.DataFrame(price_history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    else:
        t = yf.Ticker(ticker)
        df = t.history(period="1y")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Trend
    sma_50 = ta.trend.sma_indicator(close, window=50)
    sma_200 = ta.trend.sma_indicator(close, window=200)
    sma50_val = round(float(sma_50.iloc[-1]), 4) if not sma_50.empty else None
    sma200_val = round(float(sma_200.iloc[-1]), 4) if not sma_200.empty else None
    current = round(float(close.iloc[-1]), 4)

    golden_cross = (sma50_val and sma200_val and sma50_val > sma200_val)
    death_cross = (sma50_val and sma200_val and sma50_val < sma200_val)

    # Momentum
    rsi = ta.momentum.rsi(close, window=14)
    rsi_val = round(float(rsi.iloc[-1]), 2) if not rsi.empty else None
    rsi_signal = (
        "overbought" if rsi_val and rsi_val > 70
        else "oversold" if rsi_val and rsi_val < 30
        else "neutral"
    )

    macd_ind = ta.trend.MACD(close)
    macd_val = round(float(macd_ind.macd().iloc[-1]), 4)
    macd_sig = round(float(macd_ind.macd_signal().iloc[-1]), 4)
    macd_hist = round(float(macd_ind.macd_diff().iloc[-1]), 4)
    macd_trend = "bullish" if macd_hist > 0 else "bearish"

    # Volatility
    bb = ta.volatility.BollingerBands(close)
    bb_upper = round(float(bb.bollinger_hband().iloc[-1]), 4)
    bb_lower = round(float(bb.bollinger_lband().iloc[-1]), 4)
    bb_width = round(float(bb.bollinger_wband().iloc[-1]), 4)
    atr = ta.volatility.average_true_range(high, low, close, window=14)
    atr_val = round(float(atr.iloc[-1]), 4)

    # Volume
    vol_sma = ta.trend.sma_indicator(volume.astype(float), window=20)
    vol_sma_val = round(float(vol_sma.iloc[-1]), 0)
    recent_vol = float(volume.iloc[-5:].mean())
    vol_trend = (
        "increasing" if recent_vol > vol_sma_val * 1.1
        else "decreasing" if recent_vol < vol_sma_val * 0.9
        else "stable"
    )

    # Overall signal
    bullish_signals = sum([
        golden_cross,
        rsi_signal == "oversold",
        macd_trend == "bullish",
        current > sma50_val if sma50_val else False,
    ])
    bearish_signals = sum([
        death_cross,
        rsi_signal == "overbought",
        macd_trend == "bearish",
        current < sma50_val if sma50_val else False,
    ])
    overall = "bullish" if bullish_signals > bearish_signals else "bearish" if bearish_signals > bullish_signals else "neutral"

    return {
        "trend": {
            "sma_50": sma50_val,
            "sma_200": sma200_val,
            "golden_cross": golden_cross,
            "death_cross": death_cross,
            "price_vs_sma50": "above" if sma50_val and current > sma50_val else "below",
            "price_vs_sma200": "above" if sma200_val and current > sma200_val else "below",
        },
        "momentum": {
            "rsi_14": rsi_val,
            "rsi_signal": rsi_signal,
            "macd": macd_val,
            "macd_signal": macd_sig,
            "macd_histogram": macd_hist,
            "macd_trend": macd_trend,
        },
        "volatility": {
            "bollinger_upper": bb_upper,
            "bollinger_lower": bb_lower,
            "bollinger_width": bb_width,
            "atr_14": atr_val,
        },
        "volume": {
            "volume_sma_20": vol_sma_val,
            "volume_trend": vol_trend,
        },
        "overall_signal": overall,
    }
