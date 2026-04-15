"""
Financial ratio calculator — pure computation, no external API calls.
"""


def _safe_div(a, b):
    try:
        return a / b if b else None
    except (TypeError, ZeroDivisionError):
        return None


def calculate_ratios(fundamentals: dict) -> dict:
    inc = fundamentals.get("income_statement", {})
    bal = fundamentals.get("balance_sheet", {})
    val = fundamentals.get("valuation", {})

    equity = bal.get("total_equity")
    assets = bal.get("total_assets")
    liabilities = bal.get("total_liabilities")
    debt = bal.get("total_debt")
    cash = bal.get("cash_and_equivalents")
    net_margin = inc.get("net_margin")
    revenue = inc.get("revenue_ttm")
    pe = val.get("pe_ratio")
    eps_growth = inc.get("eps_growth_yoy")
    current_price_proxy = None  # Not computed here; requires market data

    net_income = _safe_div(revenue * net_margin, 1) if revenue and net_margin else None
    roe = _safe_div(net_income, equity)
    roa = _safe_div(net_income, assets)

    net_debt = (debt or 0) - (cash or 0)

    profitability = {
        "roe": roe,
        "roa": roa,
        "roic": None,  # Requires NOPAT — placeholder
    }

    leverage = {
        "debt_to_equity": _safe_div(debt, equity),
        "debt_to_assets": _safe_div(debt, assets),
        "net_debt_to_equity": _safe_div(net_debt, equity),
        "interest_coverage": None,  # Requires EBIT / interest expense — needs raw financials
    }

    liquidity = {
        "current_ratio": None,  # Requires current assets/liabilities — supplement from FMP
        "quick_ratio": None,
    }

    valuation_derived = {
        "earnings_yield": _safe_div(1, pe) if pe else None,
        "peg_ratio": _safe_div(pe, (eps_growth * 100)) if pe and eps_growth else None,
        "fcf_yield": None,  # Requires FCF and market cap — supplement when available
    }

    # Altman Z-score placeholder (requires additional balance sheet line items)
    risk = {"altman_z_score": None}

    return {
        "profitability": profitability,
        "leverage": leverage,
        "liquidity": liquidity,
        "valuation_derived": valuation_derived,
        "risk": risk,
    }
