"""
Fundamentals tool — yfinance primary, FMP API supplement.

yfinance provides TTM/snapshot data from the info dict and quarterly
financial statements. FMP fills gaps (annual revenue history, EBITDA,
free cash flow) when an API key is present.
"""
import requests
import yfinance as yf

from src.config import settings


# ── FMP helper ────────────────────────────────────────────────────────────────

def _fmp_get(endpoint: str, params: dict = None) -> dict | list:
    """Call FMP API and return parsed JSON. Raises on HTTP error."""
    base = "https://financialmodelingprep.com/api/v3"
    p = params or {}
    p["apikey"] = settings.FMP_API_KEY
    resp = requests.get(f"{base}{endpoint}", params=p, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _fmp_first(endpoint: str, params: dict = None) -> dict:
    """Return the first element of a FMP list response, or {}."""
    try:
        data = _fmp_get(endpoint, params)
        return data[0] if isinstance(data, list) and data else (data or {})
    except Exception:
        return {}


# ── internal helpers ──────────────────────────────────────────────────────────

def _pct(val) -> float | None:
    """Convert a margin/ratio already in decimal form to a rounded float."""
    try:
        return round(float(val), 6)
    except (TypeError, ValueError):
        return None


def _dollars(val) -> int | None:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _f(val, digits: int = 4) -> float | None:
    try:
        return round(float(val), digits)
    except (TypeError, ValueError):
        return None


def _annual_history(statements, value_key: str) -> list[dict]:
    """
    Pull a time-series of annual values from a yfinance financial DataFrame.

    statements : pd.DataFrame  — columns are date periods, index is line items
    value_key  : str           — exact row label in the DataFrame index
    """
    history = []
    if statements is None or statements.empty:
        return history
    if value_key not in statements.index:
        return history
    row = statements.loc[value_key]
    for period, val in row.items():
        try:
            history.append({
                "period": str(period.date()) if hasattr(period, "date") else str(period),
                "value": int(float(val)),
            })
        except (TypeError, ValueError):
            pass
    return sorted(history, key=lambda x: x["period"])


# ── main function ─────────────────────────────────────────────────────────────

def get_fundamentals(ticker: str) -> dict:
    """
    Fetch fundamental financial data for a ticker.

    Returns:
        {
            "income_statement": {
                "revenue_ttm": int | None,
                "revenue_growth_yoy": float | None,   # e.g. 0.08 = 8%
                "gross_profit_ttm": int | None,
                "gross_margin": float | None,
                "operating_income_ttm": int | None,
                "operating_margin": float | None,
                "net_income_ttm": int | None,
                "net_margin": float | None,
                "ebitda_ttm": int | None,
                "eps_ttm": float | None,
                "eps_diluted_ttm": float | None,
                "eps_growth_yoy": float | None,
                "shares_outstanding": int | None,
                "revenue_history": list[dict],         # [{period, value}] last 4 annual
            },
            "balance_sheet": {
                "total_assets": int | None,
                "total_liabilities": int | None,
                "total_equity": int | None,
                "cash_and_equivalents": int | None,
                "short_term_investments": int | None,
                "total_cash": int | None,              # cash + short-term investments
                "total_debt": int | None,
                "net_debt": int | None,                # total_debt - total_cash
                "working_capital": int | None,
                "book_value_per_share": float | None,
                "current_ratio": float | None,
                "quick_ratio": float | None,
            },
            "cash_flow": {
                "operating_cash_flow_ttm": int | None,
                "capex_ttm": int | None,
                "free_cash_flow_ttm": int | None,
                "fcf_margin": float | None,
            },
            "valuation": {
                "pe_ratio": float | None,
                "forward_pe": float | None,
                "pb_ratio": float | None,
                "ps_ratio": float | None,
                "ev_ebitda": float | None,
                "ev_revenue": float | None,
                "peg_ratio": float | None,
                "dividend_yield": float | None,
                "payout_ratio": float | None,
                "enterprise_value": int | None,
            },
        }

    Raises:
        ValueError: if the ticker is invalid or returns no fundamental data.
    """
    ticker = ticker.strip().upper()
    t = yf.Ticker(ticker)

    try:
        info = t.info
    except Exception as exc:
        raise ValueError(f"Failed to fetch data for '{ticker}': {exc}") from exc

    if not info or not info.get("longName"):
        raise ValueError(f"Ticker '{ticker}' not found or no fundamental data available.")

    # ── income statement ──────────────────────────────────────────────────────
    revenue_ttm = _dollars(info.get("totalRevenue"))
    net_income_ttm = _dollars(info.get("netIncomeToCommon"))
    ebitda_ttm = _dollars(info.get("ebitda"))
    gross_profit_ttm = _dollars(info.get("grossProfits"))
    operating_income_ttm = None  # derived below if possible

    # Try to pull from yfinance annual income statement DataFrame
    try:
        annual_inc = t.income_stmt           # columns = fiscal year end dates
    except Exception:
        annual_inc = None

    rev_history = _annual_history(annual_inc, "Total Revenue")

    # Derive operating income from statement if available
    if annual_inc is not None and not annual_inc.empty:
        op_labels = ["Operating Income", "EBIT"]
        for lbl in op_labels:
            if lbl in annual_inc.index:
                try:
                    operating_income_ttm = int(float(annual_inc.loc[lbl].iloc[0]))
                except (TypeError, ValueError):
                    pass
                break

    gross_margin = _pct(info.get("grossMargins"))
    operating_margin = _pct(info.get("operatingMargins"))
    net_margin = _pct(info.get("profitMargins"))

    # Compute FCF margin from ttm values
    operating_cf = _dollars(info.get("operatingCashflow"))
    capex = _dollars(info.get("capitalExpenditures"))
    if capex is not None and capex > 0:
        # yfinance returns capex as positive; FCF = OCF - capex
        free_cash_flow = _dollars(operating_cf - capex) if operating_cf is not None else None
    elif capex is not None and capex < 0:
        # some versions return capex as negative
        free_cash_flow = _dollars(operating_cf + capex) if operating_cf is not None else None
    else:
        free_cash_flow = _dollars(info.get("freeCashflow"))

    fcf_margin = (
        round(free_cash_flow / revenue_ttm, 6)
        if free_cash_flow is not None and revenue_ttm
        else None
    )

    income_statement = {
        "revenue_ttm": revenue_ttm,
        "revenue_growth_yoy": _pct(info.get("revenueGrowth")),
        "gross_profit_ttm": gross_profit_ttm,
        "gross_margin": gross_margin,
        "operating_income_ttm": operating_income_ttm,
        "operating_margin": operating_margin,
        "net_income_ttm": net_income_ttm,
        "net_margin": net_margin,
        "ebitda_ttm": ebitda_ttm,
        "eps_ttm": _f(info.get("trailingEps")),
        "eps_diluted_ttm": _f(info.get("trailingEps")),   # yfinance doesn't split basic/diluted
        "eps_growth_yoy": _pct(info.get("earningsGrowth")),
        "shares_outstanding": _dollars(info.get("sharesOutstanding")),
        "revenue_history": rev_history,
    }

    # ── balance sheet ─────────────────────────────────────────────────────────
    total_assets = _dollars(info.get("totalAssets"))
    total_liab = _dollars(info.get("totalLiab"))
    total_equity = _dollars(info.get("totalStockholderEquity"))
    cash_eq = _dollars(info.get("cash"))
    short_inv = _dollars(info.get("shortTermInvestments"))
    total_cash = _dollars(info.get("totalCash"))   # yfinance already sums cash + ST investments
    total_debt = _dollars(info.get("totalDebt"))

    net_debt = (
        _dollars(total_debt - total_cash)
        if total_debt is not None and total_cash is not None
        else None
    )

    # working capital from yfinance balance sheet DataFrame
    working_capital = _dollars(info.get("totalCurrentAssets", 0) - info.get("totalCurrentLiabilities", 0)) or None
    book_value_per_share = _f(info.get("bookValue"))
    current_ratio = _f(info.get("currentRatio"))
    quick_ratio = _f(info.get("quickRatio"))

    balance_sheet = {
        "total_assets": total_assets,
        "total_liabilities": total_liab,
        "total_equity": total_equity,
        "cash_and_equivalents": cash_eq,
        "short_term_investments": short_inv,
        "total_cash": total_cash,
        "total_debt": total_debt,
        "net_debt": net_debt,
        "working_capital": working_capital,
        "book_value_per_share": book_value_per_share,
        "current_ratio": current_ratio,
        "quick_ratio": quick_ratio,
    }

    # ── cash flow ─────────────────────────────────────────────────────────────
    cash_flow = {
        "operating_cash_flow_ttm": operating_cf,
        "capex_ttm": capex,
        "free_cash_flow_ttm": free_cash_flow,
        "fcf_margin": fcf_margin,
    }

    # ── valuation ─────────────────────────────────────────────────────────────
    valuation = {
        "pe_ratio": _f(info.get("trailingPE")),
        "forward_pe": _f(info.get("forwardPE")),
        "pb_ratio": _f(info.get("priceToBook")),
        "ps_ratio": _f(info.get("priceToSalesTrailing12Months")),
        "ev_ebitda": _f(info.get("enterpriseToEbitda")),
        "ev_revenue": _f(info.get("enterpriseToRevenue")),
        "peg_ratio": _f(info.get("pegRatio")),
        "dividend_yield": _pct(info.get("dividendYield")),
        "payout_ratio": _pct(info.get("payoutRatio")),
        "enterprise_value": _dollars(info.get("enterpriseValue")),
    }

    # ── FMP supplement ────────────────────────────────────────────────────────
    # Fill any None values that FMP can provide when the key is configured
    if settings.FMP_API_KEY:
        _fmp_supplement(ticker, income_statement, balance_sheet, cash_flow, valuation)

    return {
        "income_statement": income_statement,
        "balance_sheet": balance_sheet,
        "cash_flow": cash_flow,
        "valuation": valuation,
    }


def _fmp_supplement(
    ticker: str,
    income: dict,
    balance: dict,
    cash_flow: dict,
    valuation: dict,
) -> None:
    """Mutates the dicts in-place, filling None fields from FMP."""

    # Income statement — TTM from FMP
    inc = _fmp_first(f"/income-statement/{ticker}", {"limit": 1, "period": "ttm"})
    if inc:
        income.setdefault("revenue_ttm", _dollars(inc.get("revenue"))) or \
            _fill_if_none(income, "revenue_ttm", _dollars(inc.get("revenue")))
        _fill_if_none(income, "gross_profit_ttm", _dollars(inc.get("grossProfit")))
        _fill_if_none(income, "operating_income_ttm", _dollars(inc.get("operatingIncome")))
        _fill_if_none(income, "net_income_ttm", _dollars(inc.get("netIncome")))
        _fill_if_none(income, "ebitda_ttm", _dollars(inc.get("ebitda")))
        _fill_if_none(income, "eps_ttm", _f(inc.get("eps")))
        _fill_if_none(income, "eps_diluted_ttm", _f(inc.get("epsdiluted")))

    # Annual revenue history (last 4 years) if not already populated
    if not income.get("revenue_history"):
        try:
            ann_list = _fmp_get(f"/income-statement/{ticker}", {"limit": 4}) or []
        except Exception:
            ann_list = []
        if isinstance(ann_list, list):
            income["revenue_history"] = sorted(
                [
                    {"period": r.get("date", ""), "value": _dollars(r.get("revenue"))}
                    for r in ann_list
                    if r.get("revenue") is not None
                ],
                key=lambda x: x["period"],
            )

    # Balance sheet
    bs = _fmp_first(f"/balance-sheet-statement/{ticker}", {"limit": 1, "period": "quarter"})
    if bs:
        _fill_if_none(balance, "total_assets", _dollars(bs.get("totalAssets")))
        _fill_if_none(balance, "total_liabilities", _dollars(bs.get("totalLiabilities")))
        _fill_if_none(balance, "total_equity", _dollars(bs.get("totalStockholdersEquity")))
        _fill_if_none(balance, "total_cash", _dollars(bs.get("cashAndShortTermInvestments")))
        _fill_if_none(balance, "total_debt", _dollars(bs.get("totalDebt")))
        _fill_if_none(balance, "working_capital", _dollars(bs.get("netReceivables")))  # proxy
        _fill_if_none(balance, "book_value_per_share", _f(bs.get("bookValuePerShare")))
        _fill_if_none(balance, "current_ratio", _f(bs.get("currentRatio")))
        _fill_if_none(balance, "quick_ratio", _f(bs.get("quickRatio")))

    # Cash flow
    cf = _fmp_first(f"/cash-flow-statement/{ticker}", {"limit": 1, "period": "ttm"})
    if cf:
        _fill_if_none(cash_flow, "operating_cash_flow_ttm", _dollars(cf.get("operatingCashFlow")))
        _fill_if_none(cash_flow, "capex_ttm", _dollars(cf.get("capitalExpenditure")))
        _fill_if_none(cash_flow, "free_cash_flow_ttm", _dollars(cf.get("freeCashFlow")))

    # Valuation
    km = _fmp_first(f"/key-metrics-ttm/{ticker}")
    if km:
        _fill_if_none(valuation, "pe_ratio", _f(km.get("peRatioTTM")))
        _fill_if_none(valuation, "pb_ratio", _f(km.get("pbRatioTTM")))
        _fill_if_none(valuation, "ps_ratio", _f(km.get("priceToSalesRatioTTM")))
        _fill_if_none(valuation, "ev_ebitda", _f(km.get("evToEbitdaTTM")))
        _fill_if_none(valuation, "ev_revenue", _f(km.get("evToSalesTTM")))
        _fill_if_none(valuation, "peg_ratio", _f(km.get("pegRatioTTM")))
        _fill_if_none(valuation, "enterprise_value", _dollars(km.get("enterpriseValueTTM")))


def _fill_if_none(d: dict, key: str, value) -> None:
    """Set d[key] = value only if d[key] is currently None."""
    if d.get(key) is None and value is not None:
        d[key] = value
