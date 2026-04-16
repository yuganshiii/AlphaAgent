"""
Comprehensive tests for macro.py.

All FRED HTTP calls are mocked. _trend and _fetch_series are also tested
as units. Error handling (timeout, HTTP error, missing key) verified.
"""
import pytest
import requests
from unittest.mock import patch, MagicMock, call


# ── mock response factory ─────────────────────────────────────────────────────

def _fred_resp(values: list, ok: bool = True):
    """Return a mock FRED response for the given float values."""
    mock = MagicMock()
    mock.raise_for_status.return_value = None if ok else MagicMock(
        side_effect=requests.HTTPError("error")
    )
    mock.json.return_value = {
        "observations": [{"value": str(v)} for v in values]
    }
    return mock


def _patch_fred(responses: list):
    """Patch requests.get to return responses in sequence."""
    return patch("src.tools.macro.requests.get", side_effect=responses)


def _all_fred(value: float = 5.0, trend_values=None):
    """Return 6 identical FRED responses (one per series)."""
    vals = trend_values or [value, value - 0.1, value - 0.2]
    return [_fred_resp(vals) for _ in range(6)]


# ── _trend unit tests ─────────────────────────────────────────────────────────

def test_trend_rising():
    from src.tools.macro import _trend
    assert _trend([5.5, 5.0, 4.5]) == "rising"


def test_trend_falling():
    from src.tools.macro import _trend
    assert _trend([4.5, 5.0, 5.5]) == "falling"


def test_trend_stable():
    from src.tools.macro import _trend
    assert _trend([5.0, 5.0]) == "stable"


def test_trend_single_value_is_unknown():
    from src.tools.macro import _trend
    assert _trend([5.0]) == "unknown"


def test_trend_empty_is_unknown():
    from src.tools.macro import _trend
    assert _trend([]) == "unknown"


# ── _fetch_series unit tests ──────────────────────────────────────────────────

def test_fetch_series_returns_empty_without_key(monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "")
    from src.tools.macro import _fetch_series
    result = _fetch_series("FEDFUNDS")
    assert result == []


@patch("src.tools.macro.requests.get")
def test_fetch_series_parses_float_values(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "key")
    mock_get.return_value = _fred_resp([5.33, 5.50, 5.25])
    from src.tools.macro import _fetch_series
    result = _fetch_series("FEDFUNDS")
    assert result == [5.33, 5.50, 5.25]


@patch("src.tools.macro.requests.get")
def test_fetch_series_skips_dot_placeholder(mock_get, monkeypatch):
    """FRED uses '.' for missing data — must be silently skipped."""
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "key")
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {
        "observations": [
            {"value": "5.33"},
            {"value": "."},       # FRED missing-data placeholder
            {"value": "5.25"},
        ]
    }
    mock_get.return_value = mock
    from src.tools.macro import _fetch_series
    result = _fetch_series("FEDFUNDS")
    assert result == [5.33, 5.25]
    assert len(result) == 2


@patch("src.tools.macro.requests.get")
def test_fetch_series_timeout_returns_empty(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "key")
    mock_get.side_effect = requests.exceptions.Timeout("timed out")
    from src.tools.macro import _fetch_series
    result = _fetch_series("FEDFUNDS")
    assert result == []


@patch("src.tools.macro.requests.get")
def test_fetch_series_http_error_returns_empty(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "key")
    mock = MagicMock()
    mock.raise_for_status.side_effect = requests.HTTPError("429 Rate Limited")
    mock_get.return_value = mock
    from src.tools.macro import _fetch_series
    result = _fetch_series("FEDFUNDS")
    assert result == []


@patch("src.tools.macro.requests.get")
def test_fetch_series_connection_error_returns_empty(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "key")
    mock_get.side_effect = requests.exceptions.ConnectionError("unreachable")
    from src.tools.macro import _fetch_series
    result = _fetch_series("FEDFUNDS")
    assert result == []


# ── get_macro_context output structure ────────────────────────────────────────

def test_output_keys_no_api_key(monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "")
    from src.tools.macro import get_macro_context
    result = get_macro_context()
    assert set(result.keys()) == {
        "fed_funds_rate", "cpi_yoy", "gdp_growth", "unemployment",
        "yield_curve_spread", "vix", "macro_summary",
    }


@patch("src.tools.macro.requests.get")
def test_output_keys_with_api_key(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "key")
    mock_get.side_effect = _all_fred()
    from src.tools.macro import get_macro_context
    result = get_macro_context()
    assert set(result.keys()) == {
        "fed_funds_rate", "cpi_yoy", "gdp_growth", "unemployment",
        "yield_curve_spread", "vix", "macro_summary",
    }


@patch("src.tools.macro.requests.get")
def test_series_entries_have_value_and_trend(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "key")
    mock_get.side_effect = _all_fred()
    from src.tools.macro import get_macro_context
    result = get_macro_context()
    for key in ("fed_funds_rate", "cpi_yoy", "gdp_growth", "unemployment"):
        assert set(result[key].keys()) == {"value", "trend"}


@patch("src.tools.macro.requests.get")
def test_yield_curve_has_inverted_flag(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "key")
    mock_get.side_effect = _all_fred()
    from src.tools.macro import get_macro_context
    yc = get_macro_context()["yield_curve_spread"]
    assert "inverted" in yc
    assert isinstance(yc["inverted"], bool)


@patch("src.tools.macro.requests.get")
def test_vix_has_level_field(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "key")
    mock_get.side_effect = _all_fred()
    from src.tools.macro import get_macro_context
    vix = get_macro_context()["vix"]
    assert "level" in vix
    assert vix["level"] in {"low", "moderate", "high", "extreme", "unknown"}


# ── VIX level classification ──────────────────────────────────────────────────

def _run_with_vix(vix_value: float, monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "key")
    responses = [_fred_resp([5.33]) for _ in range(5)]   # 5 non-vix series
    responses.append(_fred_resp([vix_value]))             # vix
    with patch("src.tools.macro.requests.get", side_effect=responses):
        from src.tools.macro import get_macro_context
        return get_macro_context()["vix"]["level"]


def test_vix_level_low(monkeypatch):
    assert _run_with_vix(12.5, monkeypatch) == "low"


def test_vix_level_moderate(monkeypatch):
    assert _run_with_vix(20.0, monkeypatch) == "moderate"


def test_vix_level_high(monkeypatch):
    assert _run_with_vix(30.0, monkeypatch) == "high"


def test_vix_level_extreme(monkeypatch):
    assert _run_with_vix(45.0, monkeypatch) == "extreme"


def test_vix_level_unknown_when_no_key(monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "")
    from src.tools.macro import get_macro_context
    assert get_macro_context()["vix"]["level"] == "unknown"


# ── yield curve inversion ─────────────────────────────────────────────────────

def _run_with_yield_curve(spread: float, monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "key")
    # yield_curve_spread is the 5th series (index 4 in SERIES dict iteration)
    responses = [_fred_resp([5.33]) for _ in range(4)]   # fed, cpi, gdp, unemployment
    responses.append(_fred_resp([spread]))                # yield_curve_spread
    responses.append(_fred_resp([18.0]))                  # vix
    with patch("src.tools.macro.requests.get", side_effect=responses):
        from src.tools.macro import get_macro_context
        return get_macro_context()["yield_curve_spread"]


def test_yield_curve_inverted_when_negative(monkeypatch):
    yc = _run_with_yield_curve(-0.5, monkeypatch)
    assert yc["inverted"] is True


def test_yield_curve_not_inverted_when_positive(monkeypatch):
    yc = _run_with_yield_curve(1.2, monkeypatch)
    assert yc["inverted"] is False


def test_yield_curve_not_inverted_when_zero(monkeypatch):
    yc = _run_with_yield_curve(0.0, monkeypatch)
    assert yc["inverted"] is False


# ── all values None without API key ──────────────────────────────────────────

def test_all_values_none_without_api_key(monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "")
    from src.tools.macro import get_macro_context
    result = get_macro_context()
    assert result["fed_funds_rate"]["value"] is None
    assert result["cpi_yoy"]["value"] is None
    assert result["gdp_growth"]["value"] is None
    assert result["unemployment"]["value"] is None


# ── correct values returned ───────────────────────────────────────────────────

@patch("src.tools.macro.requests.get")
def test_fed_funds_rate_value_correct(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "key")
    mock_get.side_effect = [
        _fred_resp([5.33, 5.50, 5.25]),  # fed — latest is 5.33
        *[_fred_resp([1.0]) for _ in range(5)],
    ]
    from src.tools.macro import get_macro_context
    result = get_macro_context()
    assert result["fed_funds_rate"]["value"] == 5.33


@patch("src.tools.macro.requests.get")
def test_fed_funds_trend_rising(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "key")
    mock_get.side_effect = [
        _fred_resp([5.5, 5.0, 4.5]),   # fed: 5.5 > 4.5 → rising
        *[_fred_resp([1.0]) for _ in range(5)],
    ]
    from src.tools.macro import get_macro_context
    assert get_macro_context()["fed_funds_rate"]["trend"] == "rising"


@patch("src.tools.macro.requests.get")
def test_all_6_series_requested(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "key")
    mock_get.side_effect = _all_fred()
    from src.tools.macro import get_macro_context
    get_macro_context()
    assert mock_get.call_count == 6


# ── macro_summary content ─────────────────────────────────────────────────────

def test_macro_summary_unavailable_without_key(monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "")
    from src.tools.macro import get_macro_context
    assert get_macro_context()["macro_summary"] == "Macro data unavailable."


@patch("src.tools.macro.requests.get")
def test_macro_summary_mentions_fed_rate(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "key")
    mock_get.side_effect = [
        _fred_resp([5.33, 5.50, 5.00]),   # fed
        *[_fred_resp([1.0]) for _ in range(5)],
    ]
    from src.tools.macro import get_macro_context
    result = get_macro_context()
    assert "5.33" in result["macro_summary"]


@patch("src.tools.macro.requests.get")
def test_macro_summary_mentions_yield_curve_inversion(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "key")
    mock_get.side_effect = [
        _fred_resp([5.33]),               # fed
        _fred_resp([310.0]),              # cpi
        _fred_resp([28_000.0]),           # gdp
        _fred_resp([3.9]),                # unemployment
        _fred_resp([-0.5]),               # yield curve — INVERTED
        _fred_resp([18.0]),               # vix
    ]
    from src.tools.macro import get_macro_context
    result = get_macro_context()
    assert "inverted" in result["macro_summary"].lower()


@patch("src.tools.macro.requests.get")
def test_macro_summary_mentions_vix(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "key")
    mock_get.side_effect = _all_fred(5.0, [5.0, 4.9, 4.8])
    from src.tools.macro import get_macro_context
    result = get_macro_context()
    assert "VIX" in result["macro_summary"] or "vix" in result["macro_summary"].lower()


# ── graceful degradation on API failure ──────────────────────────────────────

@patch("src.tools.macro.requests.get")
def test_single_series_timeout_does_not_crash(mock_get, monkeypatch):
    """One series timing out should not crash the whole function."""
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "key")
    # First series (fed) times out; rest succeed
    mock_get.side_effect = [
        requests.exceptions.Timeout("timed out"),
        *[_fred_resp([5.0]) for _ in range(5)],
    ]
    from src.tools.macro import get_macro_context
    result = get_macro_context()   # must not raise
    assert result["fed_funds_rate"]["value"] is None


@patch("src.tools.macro.requests.get")
def test_all_series_timeout_returns_empty_values(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "key")
    mock_get.side_effect = [requests.exceptions.Timeout("timeout")] * 6
    from src.tools.macro import get_macro_context
    result = get_macro_context()
    assert result["fed_funds_rate"]["value"] is None
    assert result["vix"]["value"] is None
    assert result["macro_summary"] == "Macro data unavailable."
