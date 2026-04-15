"""
Fundamentals tool — yfinance primary, FMP API for deeper data.
"""
import requests
import yfinance as yf

from src.config import settings


def _fmp(endpoint: str, params: dict = None) -> dict:
    base = "https://financialmodelingprep.com/api/v3"
    params = params or {}
    params["apikey"] = settings.FMP_API_KEY
    resp = requests.get(f"{base}{endpoint}", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data[0] if isinstance(data, list) and data else data


def get_fundamentals(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.info

    # Income statement
    income = {
        "revenue_ttm": info.get("totalRevenue"),
        "revenue_growth_yoy": info.get("revenueGrowth"),
        "gross_margin": info.get("grossMargins"),
        "operating_margin": info.get("operatingMargins"),
        "net_margin": info.get("profitMargins"),
        "eps_ttm": info.get("trailingEps"),
        "eps_growth_yoy": info.get("earningsGrowth"),
    }

    # Balance sheet
    balance = {
        "total_assets": info.get("totalAssets"),
        "total_liabilities": info.get("totalLiab"),
        "total_equity": info.get("totalStockholderEquity"),
        "cash_and_equivalents": info.get("totalCash"),
        "total_debt": info.get("totalDebt"),
    }

    # Valuation
    valuation = {
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "pb_ratio": info.get("priceToBook"),
        "ps_ratio": info.get("priceToSalesTrailing12Months"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "dividend_yield": info.get("dividendYield"),
    }

    # Supplement with FMP if key is available
    if settings.FMP_API_KEY:
        try:
            fmp_data = _fmp(f"/income-statement/{ticker}", {"limit": 1})
            income.setdefault("revenue_ttm", fmp_data.get("revenue"))
            income.setdefault("eps_ttm", fmp_data.get("eps"))
        except Exception:
            pass

    return {
        "income_statement": income,
        "balance_sheet": balance,
        "valuation": valuation,
    }
