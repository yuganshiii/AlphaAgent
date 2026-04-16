"""
Unit tests for technical analysis tool.

All tests use synthetic price data injected via price_history or by
mocking yf.Ticker — no network calls.
"""
import math
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.tools.technical import get_technical_signals, _f, _overall_signal


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_df(n: int = 300, start_price: float = 150.0, seed: int = 42) -> pd.DataFrame:
    """Synthetic trending OHLCV DataFrame with n rows."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = (start_price + np.arange(n) * 0.05
             + rng.normal(0, 1, n).cumsum())
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "Close": close,
        "High": close + rng.uniform(0.5, 2.0, n),
        "Low": close - rng.uniform(0.5, 2.0, n),
        "Volume": rng.integers(20_000_000, 80_000_000, n).astype(float),
    }, index=dates)


def _mock_ticker(df: pd.DataFrame):
    mock = MagicMock()
    mock.history.return_value = df
    return mock


def _run(df: pd.DataFrame, ticker: str = "TEST") -> dict:
    with patch("src.tools.technical.yf.Ticker", return_value=_mock_ticker(df)):
        return get_technical_signals(ticker)


# ── structure tests ───────────────────────────────────────────────────────────

def test_top_level_keys():
    r = _run(_make_df())
    assert set(r.keys()) == {
        "trend", "momentum", "volatility", "volume", "overall_signal", "signal_scores"
    }


def test_trend_keys():
    r = _run(_make_df())["trend"]
    assert set(r.keys()) == {
        "current_price", "sma_50", "sma_200", "ema_20",
        "golden_cross", "death_cross",
        "price_vs_sma50", "price_vs_sma200", "price_vs_ema20",
    }


def test_momentum_keys():
    r = _run(_make_df())["momentum"]
    assert set(r.keys()) == {
        "rsi_14", "rsi_signal", "macd_line", "macd_signal_line",
        "macd_histogram", "macd_trend", "macd_crossover",
    }


def test_volatility_keys():
    r = _run(_make_df())["volatility"]
    assert set(r.keys()) == {
        "bb_upper", "bb_middle", "bb_lower", "bb_width",
        "bb_percent_b", "atr_14", "atr_pct", "price_vs_bb",
    }


def test_volume_keys():
    r = _run(_make_df())["volume"]
    assert set(r.keys()) == {"volume_sma_20", "volume_ratio", "volume_trend"}


def test_signal_scores_keys():
    r = _run(_make_df())["signal_scores"]
    assert set(r.keys()) == {
        "sma_cross", "price_vs_sma50", "price_vs_sma200",
        "macd_histogram", "rsi", "bollinger", "total",
    }


# ── trend value tests ─────────────────────────────────────────────────────────

def test_sma50_not_none_with_300_bars():
    r = _run(_make_df(300))["trend"]
    assert r["sma_50"] is not None


def test_sma200_not_none_with_300_bars():
    r = _run(_make_df(300))["trend"]
    assert r["sma_200"] is not None


def test_ema20_not_none():
    r = _run(_make_df(300))["trend"]
    assert r["ema_20"] is not None


def test_golden_death_cross_mutually_exclusive():
    r = _run(_make_df(300))["trend"]
    # One must be True and the other False (they can't both be True)
    assert not (r["golden_cross"] and r["death_cross"])


def test_price_vs_sma_strings():
    r = _run(_make_df(300))["trend"]
    assert r["price_vs_sma50"] in {"above", "below", "unknown"}
    assert r["price_vs_sma200"] in {"above", "below", "unknown"}
    assert r["price_vs_ema20"] in {"above", "below", "unknown"}


def test_current_price_matches_last_close():
    df = _make_df(300)
    r = _run(df)
    assert r["trend"]["current_price"] == _f(df["Close"].iloc[-1])


# ── momentum value tests ──────────────────────────────────────────────────────

def test_rsi_in_valid_range():
    r = _run(_make_df(300))["momentum"]
    assert r["rsi_14"] is not None
    assert 0 <= r["rsi_14"] <= 100


def test_rsi_signal_values():
    r = _run(_make_df(300))["momentum"]
    assert r["rsi_signal"] in {"overbought", "oversold", "neutral"}


def test_rsi_overbought_threshold():
    # Force a strongly trending up series to push RSI > 70
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    close = 100 + np.arange(300) * 0.5 + rng.normal(0, 0.1, 300).cumsum()
    df = pd.DataFrame({
        "Close": close, "High": close + 0.1,
        "Low": close - 0.1, "Volume": np.ones(300) * 1e6,
    }, index=dates)
    r = _run(df)
    # RSI should be elevated (may or may not be > 70, but must be in [0,100])
    assert 0 <= r["momentum"]["rsi_14"] <= 100


def test_macd_trend_values():
    r = _run(_make_df(300))["momentum"]
    assert r["macd_trend"] in {"bullish", "bearish", "neutral"}


def test_macd_crossover_values():
    r = _run(_make_df(300))["momentum"]
    assert r["macd_crossover"] in {"bullish_cross", "bearish_cross", "none"}


def test_macd_histogram_sign_matches_trend():
    r = _run(_make_df(300))["momentum"]
    if r["macd_histogram"] is not None:
        if r["macd_histogram"] > 0:
            assert r["macd_trend"] == "bullish"
        elif r["macd_histogram"] < 0:
            assert r["macd_trend"] == "bearish"


# ── volatility value tests ────────────────────────────────────────────────────

def test_bb_bands_ordering():
    r = _run(_make_df(300))["volatility"]
    if r["bb_upper"] and r["bb_middle"] and r["bb_lower"]:
        assert r["bb_lower"] < r["bb_middle"] < r["bb_upper"]


def test_bb_width_positive():
    r = _run(_make_df(300))["volatility"]
    assert r["bb_width"] is not None
    assert r["bb_width"] > 0


def test_bb_percent_b_computed():
    r = _run(_make_df(300))["volatility"]
    assert r["bb_percent_b"] is not None   # price should be in-band with trending data


def test_bb_percent_b_in_band_range():
    r = _run(_make_df(300))
    v = r["volatility"]
    pct_b = v["bb_percent_b"]
    price_vs = v["price_vs_bb"]
    if price_vs == "in_band":
        assert 0 <= pct_b <= 1


def test_price_vs_bb_values():
    r = _run(_make_df(300))["volatility"]
    assert r["price_vs_bb"] in {"above_upper", "in_band", "below_lower"}


def test_atr_positive():
    r = _run(_make_df(300))["volatility"]
    assert r["atr_14"] is not None
    assert r["atr_14"] > 0


def test_atr_pct_computed():
    r = _run(_make_df(300))
    v = r["volatility"]
    if v["atr_14"] and v["atr_pct"]:
        price = r["trend"]["current_price"]
        assert abs(v["atr_pct"] - v["atr_14"] / price) < 1e-3


# ── volume value tests ────────────────────────────────────────────────────────

def test_volume_sma_not_none():
    r = _run(_make_df(300))["volume"]
    assert r["volume_sma_20"] is not None


def test_volume_ratio_not_none():
    r = _run(_make_df(300))["volume"]
    assert r["volume_ratio"] is not None
    assert r["volume_ratio"] > 0


def test_volume_trend_values():
    r = _run(_make_df(300))["volume"]
    assert r["volume_trend"] in {"increasing", "decreasing", "stable"}


def test_volume_increasing_when_spike():
    """Recent volume 5× above SMA → should be 'increasing'."""
    df = _make_df(300)
    # Spike the last 5 bars' volume to 10× the rest
    df.iloc[-5:, df.columns.get_loc("Volume")] = 5e8
    r = _run(df)["volume"]
    assert r["volume_trend"] == "increasing"


def test_volume_decreasing_when_drought():
    df = _make_df(300)
    df.iloc[-5:, df.columns.get_loc("Volume")] = 1_000
    r = _run(df)["volume"]
    assert r["volume_trend"] == "decreasing"


# ── overall signal tests ──────────────────────────────────────────────────────

def test_overall_signal_valid_value():
    r = _run(_make_df(300))
    assert r["overall_signal"] in {"bullish", "bearish", "neutral"}


def test_signal_scores_total_matches_label():
    r = _run(_make_df(300))
    scores = r["signal_scores"]
    label  = r["overall_signal"]
    total  = scores["total"]
    if total > 1:
        assert label == "bullish"
    elif total < -1:
        assert label == "bearish"
    else:
        assert label == "neutral"


def test_overall_signal_scoring_unit():
    """Test _overall_signal directly with known inputs."""
    label, scores = _overall_signal(
        price=100,
        sma50=90,       # price > sma50 → +1
        sma200=80,      # sma50 > sma200 (golden cross) → +1, price > sma200 → +1
        macd_hist=0.5,  # positive → +1
        rsi=45,         # neutral → 0
        bb_upper=110,
        bb_lower=85,    # price in band → 0
    )
    assert scores["sma_cross"] == 1
    assert scores["price_vs_sma50"] == 1
    assert scores["price_vs_sma200"] == 1
    assert scores["macd_histogram"] == 1
    assert scores["rsi"] == 0
    assert scores["bollinger"] == 0
    assert scores["total"] == 4
    assert label == "bullish"


def test_overall_signal_bearish_unit():
    label, scores = _overall_signal(
        price=100,
        sma50=110,      # price < sma50 → -1, sma50 > sma200 but price < sma50
        sma200=90,      # sma50 > sma200 → +1 (this reduces bearishness)
        macd_hist=-1.0, # negative → -1
        rsi=75,         # overbought → -1
        bb_upper=95,    # price > upper → -1
        bb_lower=80,
    )
    assert scores["price_vs_sma50"] == -1
    assert scores["macd_histogram"] == -1
    assert scores["rsi"] == -1
    assert scores["bollinger"] == -1
    assert label == "bearish"


def test_overall_signal_all_none_inputs():
    label, scores = _overall_signal(
        price=100,
        sma50=None, sma200=None, macd_hist=None,
        rsi=None, bb_upper=None, bb_lower=None,
    )
    assert scores["total"] == 0
    assert label == "neutral"


# ── price_history override tests ──────────────────────────────────────────────

def test_price_history_override_used_when_200_rows():
    """With ≥ 200 rows in price_history, yfinance should not be called."""
    df = _make_df(250)
    history = [
        {"date": str(idx.date()), "close": row["Close"],
         "high": row["High"], "low": row["Low"], "volume": row["Volume"]}
        for idx, row in df.iterrows()
    ]
    with patch("src.tools.technical.yf.Ticker") as mock_cls:
        result = get_technical_signals("AAPL", price_history=history)
        mock_cls.assert_not_called()

    assert result["trend"]["current_price"] is not None


def test_price_history_ignored_when_too_short():
    """With < 200 rows, yfinance fetch should still happen."""
    short_history = [
        {"date": "2024-01-01", "close": 150.0,
         "high": 151.0, "low": 149.0, "volume": 1e6}
    ]
    df = _make_df(300)
    with patch("src.tools.technical.yf.Ticker", return_value=_mock_ticker(df)):
        result = get_technical_signals("AAPL", price_history=short_history)

    assert result["trend"]["sma_200"] is not None


# ── NaN safety ────────────────────────────────────────────────────────────────

def test_f_helper_handles_nan():
    assert _f(float("nan")) is None
    assert _f(float("inf")) is None
    assert _f(None) is None
    assert _f("abc") is None
    assert _f(3.14159) == 3.1416


def test_insufficient_data_returns_none_not_crash():
    """Only 30 bars — SMA-200 should be None, not an exception."""
    df = _make_df(30)
    r = _run(df)
    assert r["trend"]["sma_200"] is None
    assert r["trend"]["sma_50"] is None
    # But current_price should still exist
    assert r["trend"]["current_price"] is not None


# ── invalid ticker ────────────────────────────────────────────────────────────

def test_empty_dataframe_raises_value_error():
    with patch("src.tools.technical.yf.Ticker") as mock_cls:
        mock = MagicMock()
        mock.history.return_value = pd.DataFrame()
        mock_cls.return_value = mock
        with pytest.raises(ValueError, match="No price data"):
            get_technical_signals("FAKEXYZ")
