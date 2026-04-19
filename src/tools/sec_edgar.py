"""
SEC EDGAR tool — filing metadata and XBRL financial facts.
No API key required; must use a descriptive User-Agent per SEC policy.
"""
import requests

from src.config import settings

BASE = "https://data.sec.gov"
HEADERS = {"User-Agent": settings.SEC_USER_AGENT}

# Ticker → CIK lookup
_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"


def _get_cik(ticker: str) -> str:
    resp = requests.get(_TICKER_CIK_URL, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry["ticker"].upper() == ticker_upper:
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"CIK not found for ticker: {ticker}")


def _index_url(cik: str, accession: str) -> str:
    """Return the EDGAR filing index page URL for a specific accession.

    Format: https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{acc}-index.htm
    This page lists all documents in the filing and links to the primary document.
    """
    cik_int = int(cik)
    acc_nodash = accession.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{acc_nodash}/{accession}-index.htm"
    )


def get_sec_filings(ticker: str) -> dict:
    cik = _get_cik(ticker)

    # Submissions
    sub_resp = requests.get(f"{BASE}/submissions/CIK{cik}.json", headers=HEADERS, timeout=15)
    sub_resp.raise_for_status()
    sub = sub_resp.json()

    company_name = sub.get("name", ticker)
    recent = sub.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])

    recent_filings = []
    latest_10k_url = None
    latest_10q_url = None

    for form, date, acc in zip(forms, dates, accessions):
        if form in ("10-K", "10-Q"):
            url = _index_url(cik, acc)
            entry = {"form": form, "date": date, "accession": acc, "url": url}
            recent_filings.append(entry)
            if form == "10-K" and not latest_10k_url:
                latest_10k_url = url
            if form == "10-Q" and not latest_10q_url:
                latest_10q_url = url
        if len(recent_filings) >= 8:
            break

    # XBRL facts
    xbrl_resp = requests.get(
        f"{BASE}/api/xbrl/companyfacts/CIK{cik}.json", headers=HEADERS, timeout=20
    )
    xbrl_highlights: dict = {}
    if xbrl_resp.ok:
        facts = xbrl_resp.json().get("facts", {}).get("us-gaap", {})

        def _extract(concept: str) -> list:
            units = facts.get(concept, {}).get("units", {})
            values = units.get("USD") or units.get("shares") or []
            annual = [v for v in values if v.get("form") == "10-K"]
            return sorted(annual, key=lambda x: x.get("end", ""))[-4:]

        xbrl_highlights = {
            "revenue_history": _extract("Revenues") or _extract("RevenueFromContractWithCustomerExcludingAssessedTax"),
            "net_income_history": _extract("NetIncomeLoss"),
            "assets_history": _extract("Assets"),
            "eps_history": _extract("EarningsPerShareBasic"),
        }

    return {
        "cik": cik,
        "company_name": company_name,
        "recent_filings": recent_filings,
        "xbrl_highlights": xbrl_highlights,
        "latest_10k_url": latest_10k_url,
        "latest_10q_url": latest_10q_url,
    }
