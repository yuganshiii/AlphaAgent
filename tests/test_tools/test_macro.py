"""Unit tests for macro tool."""
from unittest.mock import patch
import pytest


def test_macro_no_key(monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "")
    from src.tools.macro import get_macro_context
    result = get_macro_context()
    assert "fed_funds_rate" in result
    assert "macro_summary" in result


@patch("src.tools.macro.requests.get")
def test_macro_with_fred(mock_get, monkeypatch):
    monkeypatch.setattr("src.tools.macro.settings.FRED_API_KEY", "test_key")
    mock_get.return_value = type("R", (), {
        "ok": True,
        "raise_for_status": lambda: None,
        "json": lambda: {"observations": [
            {"value": "5.33"},
            {"value": "5.50"},
            {"value": "5.25"},
        ]},
    })()

    from src.tools.macro import get_macro_context
    result = get_macro_context()
    assert result["fed_funds_rate"]["value"] == 5.33
