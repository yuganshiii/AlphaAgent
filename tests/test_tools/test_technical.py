"""Unit tests for technical analysis tool."""
import pytest


def test_technical_overall_signal_keys():
    """Verify output schema without hitting yfinance."""
    from unittest.mock import patch, MagicMock
    import pandas as pd
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    rng = np.random.default_rng(42)
    close = pd.Series(rng.normal(150, 5, 300).cumsum() + 100, index=dates)
    df = pd.DataFrame({
        "Close": close,
        "High": close + 2,
        "Low": close - 2,
        "Volume": rng.integers(1_000_000, 5_000_000, 300).astype(float),
    })

    with patch("src.tools.technical.yf.Ticker") as mock_cls:
        mock_t = MagicMock()
        mock_t.history.return_value = df
        mock_cls.return_value = mock_t

        from src.tools.technical import get_technical_signals
        result = get_technical_signals("AAPL")

    assert "trend" in result
    assert "momentum" in result
    assert "volatility" in result
    assert "overall_signal" in result
    assert result["overall_signal"] in {"bullish", "bearish", "neutral"}
