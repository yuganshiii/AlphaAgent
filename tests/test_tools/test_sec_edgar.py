"""Unit tests for SEC EDGAR tool."""
from unittest.mock import patch
import pytest


@patch("src.tools.sec_edgar.requests.get")
def test_get_sec_filings_structure(mock_get):
    import json

    # Mock CIK lookup
    cik_data = {"0": {"ticker": "AAPL", "cik_str": 320193}}
    sub_data = {
        "name": "Apple Inc.",
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q", "8-K"],
                "filingDate": ["2024-10-31", "2024-08-01", "2024-07-01"],
                "accessionNumber": ["0000320193-24-000123", "0000320193-24-000100", "0000320193-24-000090"],
            }
        },
    }
    xbrl_data = {"facts": {"us-gaap": {}}}

    responses = [
        type("R", (), {"ok": True, "json": lambda s=cik_data: s, "raise_for_status": lambda: None})(),
        type("R", (), {"ok": True, "json": lambda s=sub_data: s, "raise_for_status": lambda: None})(),
        type("R", (), {"ok": True, "json": lambda s=xbrl_data: s, "raise_for_status": lambda: None})(),
    ]
    mock_get.side_effect = responses

    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")

    assert result["cik"] == "0000320193"
    assert result["company_name"] == "Apple Inc."
    assert len(result["recent_filings"]) > 0
