"""
Financial ratio calculator — pure computation, no external API calls.

Takes the dict returned by get_fundamentals() and optionally the dict
returned by get_market_data(). Derives all ratios from those inputs.

Altman Z-score and FCF yield require market_data for market cap.
When market_data is absent those fields return None.
"""

# ── Assumed US corporate tax rate for NOPAT approximation ─────────────────────
_ASSUMED_TAX_RATE = 0.21


# ── safe arithmetic helpers ───────────────────────────────────────────────────

def _div(numerator, denominator, digits: int = 4) -> float | None:
    """Return numerator / denominator rounded to digits, or None on bad input."""
    try:
        if denominator == 0:
            return None
        return round(float(numerator) / float(denominator), digits)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _val(v, digits: int = 4) -> float | None:
    try:
        return round(float(v), digits)
    except (TypeError, ValueError):
        return None


# ── Altman Z-score ─────────────────────────────────────────────────────────────
#
# Original (1968) model for publicly traded manufacturers:
#   Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
#
#   X1 = Working Capital / Total Assets
#   X2 = Retained Earnings / Total Assets
#        → not available from our data; set to 0 (conservative underestimate)
#   X3 = EBIT / Total Assets  (use operating_income_ttm as proxy)
#   X4 = Market Value of Equity / Book Value of Total Liabilities
#        → requires market_cap from market_data
#   X5 = Revenue / Total Assets
#
# Interpretation:
#   Z > 2.99  → Safe zone
#   1.81–2.99 → Grey zone
#   Z < 1.81  → Distress zone

def _altman_z(
    working_capital: int | None,
    total_assets: int | None,
    ebit: int | None,
    revenue: int | None,
    total_liabilities: int | None,
    market_cap: int | None,
) -> dict:
    """
    Compute Altman Z-score and return a dict with the score and its
    interpretation. Returns all None when required inputs are missing.
    """
    missing = []
    if working_capital is None: missing.append("working_capital")
    if total_assets is None or total_assets == 0: missing.append("total_assets")
    if ebit is None: missing.append("ebit/operating_income")
    if revenue is None: missing.append("revenue")
    if total_liabilities is None or total_liabilities == 0: missing.append("total_liabilities")
    if market_cap is None: missing.append("market_cap (pass market_data)")

    if missing:
        return {
            "score": None,
            "zone": None,
            "x1_wc_to_assets": None,
            "x2_re_to_assets": None,
            "x3_ebit_to_assets": None,
            "x4_mve_to_liabilities": None,
            "x5_revenue_to_assets": None,
            "note": f"Cannot compute — missing: {', '.join(missing)}",
        }

    x1 = working_capital / total_assets
    x2 = 0.0   # retained earnings unavailable; conservative
    x3 = ebit / total_assets
    x4 = market_cap / total_liabilities
    x5 = revenue / total_assets

    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    if z > 2.99:
        zone = "safe"
    elif z >= 1.81:
        zone = "grey"
    else:
        zone = "distress"

    return {
        "score": round(z, 4),
        "zone": zone,
        "x1_wc_to_assets": round(x1, 6),
        "x2_re_to_assets": round(x2, 6),
        "x3_ebit_to_assets": round(x3, 6),
        "x4_mve_to_liabilities": round(x4, 6),
        "x5_revenue_to_assets": round(x5, 6),
        "note": "X2 (retained earnings) set to 0 — not available from source data.",
    }


# ── main function ─────────────────────────────────────────────────────────────

def calculate_ratios(
    fundamentals: dict,
    market_data: dict | None = None,
) -> dict:
    """
    Derive financial ratios from a fundamentals dict (and optional market_data).

    Args:
        fundamentals:  dict returned by get_fundamentals()
        market_data:   dict returned by get_market_data() — needed for
                       FCF yield and Altman Z-score (both require market_cap)

    Returns:
        {
            "profitability": {
                "roe": float | None,             # Net Income / Equity
                "roa": float | None,             # Net Income / Assets
                "roic": float | None,            # NOPAT / Invested Capital
                "gross_margin": float | None,    # passed through from fundamentals
                "operating_margin": float | None,
                "net_margin": float | None,
                "ebitda_margin": float | None,   # EBITDA / Revenue
                "asset_turnover": float | None,  # Revenue / Assets
            },
            "leverage": {
                "debt_to_equity": float | None,
                "debt_to_assets": float | None,
                "net_debt_to_equity": float | None,
                "net_debt_to_ebitda": float | None,
                "equity_multiplier": float | None,  # Assets / Equity
                "debt_to_ebitda": float | None,
            },
            "liquidity": {
                "current_ratio": float | None,
                "quick_ratio": float | None,
                "cash_ratio": float | None,       # Cash / Current Liabilities
                "working_capital": int | None,    # absolute $ amount
            },
            "valuation_derived": {
                "earnings_yield": float | None,   # 1 / PE
                "fcf_yield": float | None,        # FCF / Market Cap (needs market_data)
                "fcf_per_share": float | None,    # FCF / Shares Outstanding
                "price_to_fcf": float | None,     # Market Cap / FCF
            },
            "risk": {
                "altman_z_score": dict,           # full Z-score breakdown
            },
        }
    """
    inc = fundamentals.get("income_statement", {}) or {}
    bal = fundamentals.get("balance_sheet", {}) or {}
    cf  = fundamentals.get("cash_flow", {}) or {}
    val = fundamentals.get("valuation", {}) or {}
    md  = market_data or {}

    # ── raw values ─────────────────────────────────────────────────────────────
    revenue         = inc.get("revenue_ttm")
    net_income      = inc.get("net_income_ttm")
    ebitda          = inc.get("ebitda_ttm")
    operating_income= inc.get("operating_income_ttm")
    shares          = inc.get("shares_outstanding")

    total_assets    = bal.get("total_assets")
    total_liab      = bal.get("total_liabilities")
    total_equity    = bal.get("total_equity")
    total_debt      = bal.get("total_debt")
    total_cash      = bal.get("total_cash")
    net_debt        = bal.get("net_debt")
    working_capital = bal.get("working_capital")
    current_ratio   = bal.get("current_ratio")
    quick_ratio     = bal.get("quick_ratio")

    fcf             = cf.get("free_cash_flow_ttm")
    market_cap      = md.get("market_cap")

    pe              = val.get("pe_ratio")

    # ── profitability ──────────────────────────────────────────────────────────
    # ROIC = NOPAT / Invested Capital
    # NOPAT ≈ Operating Income × (1 - assumed tax rate)
    # Invested Capital = Total Equity + Total Debt - Cash
    nopat = (
        operating_income * (1 - _ASSUMED_TAX_RATE)
        if operating_income is not None
        else None
    )
    invested_capital = (
        (total_equity or 0) + (total_debt or 0) - (total_cash or 0)
        if total_equity is not None or total_debt is not None
        else None
    )
    roic = _div(nopat, invested_capital)
    ebitda_margin = _div(ebitda, revenue)

    profitability = {
        "roe": _div(net_income, total_equity),
        "roa": _div(net_income, total_assets),
        "roic": roic,
        "gross_margin": _val(inc.get("gross_margin")),
        "operating_margin": _val(inc.get("operating_margin")),
        "net_margin": _val(inc.get("net_margin")),
        "ebitda_margin": ebitda_margin,
        "asset_turnover": _div(revenue, total_assets),
    }

    # ── leverage ───────────────────────────────────────────────────────────────
    leverage = {
        "debt_to_equity": _div(total_debt, total_equity),
        "debt_to_assets": _div(total_debt, total_assets),
        "net_debt_to_equity": _div(net_debt, total_equity),
        "net_debt_to_ebitda": _div(net_debt, ebitda),
        "equity_multiplier": _div(total_assets, total_equity),
        "debt_to_ebitda": _div(total_debt, ebitda),
    }

    # ── liquidity ──────────────────────────────────────────────────────────────
    # cash_ratio = cash / current_liabilities
    # We can derive current_liabilities if we have working_capital and current_assets,
    # but those aren't available separately. Approximate: cash / (current_assets = WC + CL)
    # is circular. Only compute cash_ratio when both current_ratio and working_capital exist.
    cash_ratio = None
    if current_ratio is not None and working_capital is not None and total_cash is not None:
        # current_liabilities = working_capital / (current_ratio - 1) when CR > 1
        if current_ratio > 1:
            current_liab_est = working_capital / (current_ratio - 1)
            cash_ratio = _div(total_cash, current_liab_est)

    liquidity = {
        "current_ratio": _val(current_ratio),
        "quick_ratio": _val(quick_ratio),
        "cash_ratio": cash_ratio,
        "working_capital": working_capital,
    }

    # ── valuation derived ──────────────────────────────────────────────────────
    fcf_yield = _div(fcf, market_cap) if market_cap else None
    fcf_per_share = _div(fcf, shares) if shares else None
    price_to_fcf = _div(market_cap, fcf) if market_cap and fcf else None

    valuation_derived = {
        "earnings_yield": _div(1, pe) if pe else None,
        "fcf_yield": fcf_yield,
        "fcf_per_share": fcf_per_share,
        "price_to_fcf": price_to_fcf,
    }

    # ── risk: Altman Z-score ───────────────────────────────────────────────────
    z_score = _altman_z(
        working_capital=working_capital,
        total_assets=total_assets,
        ebit=operating_income,
        revenue=revenue,
        total_liabilities=total_liab,
        market_cap=market_cap,
    )

    return {
        "profitability": profitability,
        "leverage": leverage,
        "liquidity": liquidity,
        "valuation_derived": valuation_derived,
        "risk": {"altman_z_score": z_score},
    }
