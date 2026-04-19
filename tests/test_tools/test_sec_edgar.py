"""
Comprehensive tests for sec_edgar.py.

All network calls are mocked via unittest.mock.patch on requests.get.
"""
import pytest
import requests
from unittest.mock import patch, MagicMock, call


# ── mock response factory ─────────────────────────────────────────────────────

def _resp(data, ok=True, status=200):
    """Create a minimal mock requests.Response."""
    mock = MagicMock()
    mock.ok = ok
    mock.status_code = status
    mock.json.return_value = data
    if not ok:
        mock.raise_for_status.side_effect = requests.HTTPError(
            f"HTTP {status}", response=mock
        )
    else:
        mock.raise_for_status.return_value = None
    return mock


# ── canonical test data ───────────────────────────────────────────────────────

CIK_DATA = {
    "0": {"ticker": "AAPL", "cik_str": 320193},
    "1": {"ticker": "MSFT", "cik_str": 789019},
}

SUBMISSIONS = {
    "name": "Apple Inc.",
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "8-K", "10-Q", "10-K", "10-Q", "10-Q", "10-K", "8-K", "10-Q"],
            "filingDate": [
                "2024-10-31", "2024-08-01", "2024-07-01",
                "2024-05-01", "2023-11-01", "2023-08-01",
                "2023-05-01", "2022-10-31", "2022-08-01", "2022-05-01",
            ],
            "accessionNumber": [
                "0000320193-24-000123", "0000320193-24-000100", "0000320193-24-000090",
                "0000320193-24-000080", "0000320193-23-000123", "0000320193-23-000100",
                "0000320193-23-000080", "0000320193-22-000123", "0000320193-22-000090",
                "0000320193-22-000080",
            ],
        }
    },
}

XBRL_FACTS = {
    "facts": {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"end": "2024-09-30", "val": 400_000_000_000, "form": "10-K"},
                        {"end": "2023-09-30", "val": 383_000_000_000, "form": "10-K"},
                        {"end": "2022-09-30", "val": 394_000_000_000, "form": "10-K"},
                        {"end": "2021-09-30", "val": 365_000_000_000, "form": "10-K"},
                        {"end": "2024-06-30", "val": 85_000_000_000, "form": "10-Q"},  # excluded
                    ]
                }
            },
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {"end": "2024-09-30", "val": 93_736_000_000, "form": "10-K"},
                        {"end": "2023-09-30", "val": 96_995_000_000, "form": "10-K"},
                    ]
                }
            },
            "Assets": {
                "units": {
                    "USD": [
                        {"end": "2024-09-30", "val": 364_980_000_000, "form": "10-K"},
                    ]
                }
            },
            "EarningsPerShareBasic": {
                "units": {
                    "shares": [
                        {"end": "2024-09-30", "val": 6.11, "form": "10-K"},
                        {"end": "2023-09-30", "val": 6.16, "form": "10-K"},
                    ]
                }
            },
        }
    }
}

XBRL_EMPTY = {"facts": {"us-gaap": {}}}


def _default_responses(xbrl=None):
    return [
        _resp(CIK_DATA),
        _resp(SUBMISSIONS),
        _resp(xbrl or XBRL_FACTS),
    ]


# ── output structure ──────────────────────────────────────────────────────────

@patch("src.tools.sec_edgar.requests.get")
def test_output_keys(mock_get):
    mock_get.side_effect = _default_responses()
    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")
    assert set(result.keys()) == {
        "cik", "company_name", "recent_filings",
        "xbrl_highlights", "latest_10k_url", "latest_10q_url",
    }


@patch("src.tools.sec_edgar.requests.get")
def test_xbrl_highlights_keys(mock_get):
    mock_get.side_effect = _default_responses()
    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")
    assert set(result["xbrl_highlights"].keys()) == {
        "revenue_history", "net_income_history", "assets_history", "eps_history"
    }


@patch("src.tools.sec_edgar.requests.get")
def test_recent_filing_entry_keys(mock_get):
    mock_get.side_effect = _default_responses()
    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")
    filing = result["recent_filings"][0]
    assert set(filing.keys()) == {"form", "date", "accession", "url"}


# ── CIK handling ──────────────────────────────────────────────────────────────

@patch("src.tools.sec_edgar.requests.get")
def test_cik_zero_padded_to_10_digits(mock_get):
    mock_get.side_effect = _default_responses()
    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")
    assert result["cik"] == "0000320193"
    assert len(result["cik"]) == 10


@patch("src.tools.sec_edgar.requests.get")
def test_company_name_from_submissions(mock_get):
    mock_get.side_effect = _default_responses()
    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")
    assert result["company_name"] == "Apple Inc."


# ── filing filtering ──────────────────────────────────────────────────────────

@patch("src.tools.sec_edgar.requests.get")
def test_only_10k_and_10q_included(mock_get):
    mock_get.side_effect = _default_responses()
    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")
    forms = [f["form"] for f in result["recent_filings"]]
    assert all(f in ("10-K", "10-Q") for f in forms)
    assert "8-K" not in forms


@patch("src.tools.sec_edgar.requests.get")
def test_max_8_recent_filings(mock_get):
    mock_get.side_effect = _default_responses()
    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")
    assert len(result["recent_filings"]) <= 8


@patch("src.tools.sec_edgar.requests.get")
def test_latest_10k_url_set(mock_get):
    mock_get.side_effect = _default_responses()
    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")
    assert result["latest_10k_url"] is not None
    # URL is now the EDGAR archive index page, not a browse-edgar search page
    assert "Archives/edgar/data" in result["latest_10k_url"]
    assert result["latest_10k_url"].endswith("-index.htm")


@patch("src.tools.sec_edgar.requests.get")
def test_latest_10q_url_set(mock_get):
    mock_get.side_effect = _default_responses()
    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")
    assert result["latest_10q_url"] is not None
    assert "Archives/edgar/data" in result["latest_10q_url"]
    assert result["latest_10q_url"].endswith("-index.htm")


@patch("src.tools.sec_edgar.requests.get")
def test_no_10k_in_filings_url_is_none(mock_get):
    no_10k = {
        "name": "TestCorp",
        "filings": {
            "recent": {
                "form": ["10-Q", "8-K", "10-Q"],
                "filingDate": ["2024-08-01", "2024-07-01", "2024-05-01"],
                "accessionNumber": ["0001-24-001", "0001-24-002", "0001-24-003"],
            }
        },
    }
    mock_get.side_effect = [_resp(CIK_DATA), _resp(no_10k), _resp(XBRL_EMPTY)]
    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")
    assert result["latest_10k_url"] is None


@patch("src.tools.sec_edgar.requests.get")
def test_empty_filings_list_no_crash(mock_get):
    empty_sub = {"name": "EmptyCorp", "filings": {"recent": {"form": [], "filingDate": [], "accessionNumber": []}}}
    mock_get.side_effect = [_resp(CIK_DATA), _resp(empty_sub), _resp(XBRL_EMPTY)]
    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")
    assert result["recent_filings"] == []
    assert result["latest_10k_url"] is None


# ── XBRL parsing ──────────────────────────────────────────────────────────────

@patch("src.tools.sec_edgar.requests.get")
def test_xbrl_revenue_parsed(mock_get):
    mock_get.side_effect = _default_responses()
    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")
    rev = result["xbrl_highlights"]["revenue_history"]
    assert len(rev) > 0
    # All entries must be annual (10-K) forms
    assert all(r.get("form") == "10-K" for r in rev)


@patch("src.tools.sec_edgar.requests.get")
def test_xbrl_revenue_sorted_by_end_date(mock_get):
    mock_get.side_effect = _default_responses()
    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")
    rev = result["xbrl_highlights"]["revenue_history"]
    dates = [r["end"] for r in rev]
    assert dates == sorted(dates)


@patch("src.tools.sec_edgar.requests.get")
def test_xbrl_max_4_annual_entries(mock_get):
    mock_get.side_effect = _default_responses()
    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")
    # Input has 4 annual revenue entries → should return all 4
    assert len(result["xbrl_highlights"]["revenue_history"]) == 4


@patch("src.tools.sec_edgar.requests.get")
def test_xbrl_quarterly_excluded(mock_get):
    mock_get.side_effect = _default_responses()
    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")
    # All entries must be 10-K (XBRL_FACTS has one 10-Q entry that should be excluded)
    for entry in result["xbrl_highlights"]["revenue_history"]:
        assert entry.get("form") == "10-K"


@patch("src.tools.sec_edgar.requests.get")
def test_xbrl_fallback_revenue_concept(mock_get):
    """When 'Revenues' is empty, falls back to RevenueFromContractWithCustomer."""
    xbrl_with_fallback = {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": []}},
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {"end": "2024-09-30", "val": 400_000_000_000, "form": "10-K"},
                        ]
                    }
                },
                "NetIncomeLoss": {"units": {"USD": []}},
                "Assets": {"units": {"USD": []}},
                "EarningsPerShareBasic": {"units": {"shares": []}},
            }
        }
    }
    mock_get.side_effect = [_resp(CIK_DATA), _resp(SUBMISSIONS), _resp(xbrl_with_fallback)]
    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")
    assert len(result["xbrl_highlights"]["revenue_history"]) == 1


@patch("src.tools.sec_edgar.requests.get")
def test_xbrl_fetch_failure_returns_empty_highlights(mock_get):
    """If XBRL endpoint returns non-ok, xbrl_highlights should be empty dict."""
    mock_get.side_effect = [
        _resp(CIK_DATA),
        _resp(SUBMISSIONS),
        _resp({}, ok=False, status=404),
    ]
    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")
    assert result["xbrl_highlights"] == {}


@patch("src.tools.sec_edgar.requests.get")
def test_xbrl_eps_uses_shares_units(mock_get):
    mock_get.side_effect = _default_responses()
    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")
    eps = result["xbrl_highlights"]["eps_history"]
    assert len(eps) == 2
    assert eps[0]["val"] == 6.16   # older entry first after sort


# ── invalid ticker / error handling ──────────────────────────────────────────

@patch("src.tools.sec_edgar.requests.get")
def test_invalid_ticker_raises_value_error(mock_get):
    # CIK data does not contain the ticker
    mock_get.return_value = _resp({"0": {"ticker": "AAPL", "cik_str": 320193}})
    from src.tools.sec_edgar import get_sec_filings
    with pytest.raises(ValueError, match="CIK not found"):
        get_sec_filings("FAKEXYZ")


@patch("src.tools.sec_edgar.requests.get")
def test_cik_lookup_http_error_raises(mock_get):
    mock_get.return_value = _resp({}, ok=False, status=503)
    from src.tools.sec_edgar import get_sec_filings
    with pytest.raises(requests.HTTPError):
        get_sec_filings("AAPL")


@patch("src.tools.sec_edgar.requests.get")
def test_cik_lookup_timeout_raises(mock_get):
    mock_get.side_effect = requests.exceptions.Timeout("timed out")
    from src.tools.sec_edgar import get_sec_filings
    with pytest.raises(requests.exceptions.Timeout):
        get_sec_filings("AAPL")


@patch("src.tools.sec_edgar.requests.get")
def test_submissions_http_error_raises(mock_get):
    mock_get.side_effect = [
        _resp(CIK_DATA),
        _resp({}, ok=False, status=404),
    ]
    from src.tools.sec_edgar import get_sec_filings
    with pytest.raises(requests.HTTPError):
        get_sec_filings("AAPL")


@patch("src.tools.sec_edgar.requests.get")
def test_submissions_timeout_raises(mock_get):
    mock_get.side_effect = [
        _resp(CIK_DATA),
        requests.exceptions.Timeout("submissions timed out"),
    ]
    from src.tools.sec_edgar import get_sec_filings
    with pytest.raises(requests.exceptions.Timeout):
        get_sec_filings("AAPL")


# ── accession number format ───────────────────────────────────────────────────

@patch("src.tools.sec_edgar.requests.get")
def test_accession_number_preserved_in_filing(mock_get):
    mock_get.side_effect = _default_responses()
    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")
    first = result["recent_filings"][0]
    assert first["accession"] == "0000320193-24-000123"


@patch("src.tools.sec_edgar.requests.get")
def test_filing_date_preserved(mock_get):
    mock_get.side_effect = _default_responses()
    from src.tools.sec_edgar import get_sec_filings
    result = get_sec_filings("AAPL")
    assert result["recent_filings"][0]["date"] == "2024-10-31"
