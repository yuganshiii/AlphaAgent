"""Unit tests for ratio_calculator — pure computation, no mocking needed."""
import pytest
from src.tools.ratio_calculator import calculate_ratios


# ── shared fixtures ───────────────────────────────────────────────────────────

FUNDAMENTALS = {
    "income_statement": {
        "revenue_ttm": 400_000_000_000,
        "net_income_ttm": 100_000_000_000,
        "ebitda_ttm": 130_000_000_000,
        "operating_income_ttm": 120_000_000_000,
        "gross_margin": 0.4375,
        "operating_margin": 0.30,
        "net_margin": 0.25,
        "eps_ttm": 6.42,
        "eps_growth_yoy": 0.10,
        "shares_outstanding": 15_400_000_000,
        "revenue_history": [],
    },
    "balance_sheet": {
        "total_assets": 350_000_000_000,
        "total_liabilities": 290_000_000_000,
        "total_equity": 60_000_000_000,
        "total_debt": 120_000_000_000,
        "total_cash": 65_000_000_000,
        "net_debt": 55_000_000_000,
        "working_capital": 10_000_000_000,
        "current_ratio": 1.15,
        "quick_ratio": 0.90,
        "book_value_per_share": 4.10,
    },
    "cash_flow": {
        "operating_cash_flow_ttm": 120_000_000_000,
        "capex_ttm": 11_000_000_000,
        "free_cash_flow_ttm": 109_000_000_000,
        "fcf_margin": 0.2725,
    },
    "valuation": {
        "pe_ratio": 29.5,
        "forward_pe": 26.0,
        "pb_ratio": 45.2,
        "ps_ratio": 7.5,
        "ev_ebitda": 24.0,
        "ev_revenue": 7.8,
        "peg_ratio": 2.9,
        "dividend_yield": 0.0055,
        "payout_ratio": 0.16,
        "enterprise_value": 3_100_000_000_000,
    },
}

MARKET_DATA = {
    "market_cap": 3_000_000_000_000,
    "current_price": 195.50,
}


# ── structure tests ───────────────────────────────────────────────────────────

def test_top_level_keys():
    result = calculate_ratios(FUNDAMENTALS)
    assert set(result.keys()) == {
        "profitability", "leverage", "liquidity", "valuation_derived", "risk"
    }


def test_profitability_keys():
    r = calculate_ratios(FUNDAMENTALS)["profitability"]
    assert set(r.keys()) == {
        "roe", "roa", "roic", "gross_margin", "operating_margin",
        "net_margin", "ebitda_margin", "asset_turnover",
    }


def test_leverage_keys():
    r = calculate_ratios(FUNDAMENTALS)["leverage"]
    assert set(r.keys()) == {
        "debt_to_equity", "debt_to_assets", "net_debt_to_equity",
        "net_debt_to_ebitda", "equity_multiplier", "debt_to_ebitda",
    }


def test_liquidity_keys():
    r = calculate_ratios(FUNDAMENTALS)["liquidity"]
    assert set(r.keys()) == {"current_ratio", "quick_ratio", "cash_ratio", "working_capital"}


def test_valuation_derived_keys():
    r = calculate_ratios(FUNDAMENTALS)["valuation_derived"]
    assert set(r.keys()) == {"earnings_yield", "fcf_yield", "fcf_per_share", "price_to_fcf"}


def test_risk_keys():
    r = calculate_ratios(FUNDAMENTALS)["risk"]
    assert "altman_z_score" in r
    z = r["altman_z_score"]
    assert set(z.keys()) == {
        "score", "zone", "x1_wc_to_assets", "x2_re_to_assets",
        "x3_ebit_to_assets", "x4_mve_to_liabilities", "x5_revenue_to_assets", "note",
    }


# ── profitability correctness ─────────────────────────────────────────────────

def test_roe():
    r = calculate_ratios(FUNDAMENTALS)
    expected = round(100e9 / 60e9, 4)
    assert r["profitability"]["roe"] == expected


def test_roa():
    r = calculate_ratios(FUNDAMENTALS)
    expected = round(100e9 / 350e9, 4)
    assert r["profitability"]["roa"] == expected


def test_roic():
    # NOPAT = 120B * (1 - 0.21) = 94.8B
    # Invested Capital = 60B + 120B - 65B = 115B
    r = calculate_ratios(FUNDAMENTALS)
    nopat = 120e9 * 0.79
    inv_cap = 60e9 + 120e9 - 65e9
    expected = round(nopat / inv_cap, 4)
    assert r["profitability"]["roic"] == expected


def test_ebitda_margin():
    r = calculate_ratios(FUNDAMENTALS)
    expected = round(130e9 / 400e9, 4)
    assert r["profitability"]["ebitda_margin"] == expected


def test_asset_turnover():
    r = calculate_ratios(FUNDAMENTALS)
    expected = round(400e9 / 350e9, 4)
    assert r["profitability"]["asset_turnover"] == expected


def test_margins_passed_through():
    r = calculate_ratios(FUNDAMENTALS)["profitability"]
    assert r["gross_margin"] == 0.4375
    assert r["operating_margin"] == 0.30
    assert r["net_margin"] == 0.25


# ── leverage correctness ──────────────────────────────────────────────────────

def test_debt_to_equity():
    r = calculate_ratios(FUNDAMENTALS)
    assert r["leverage"]["debt_to_equity"] == round(120e9 / 60e9, 4)


def test_debt_to_assets():
    r = calculate_ratios(FUNDAMENTALS)
    assert r["leverage"]["debt_to_assets"] == round(120e9 / 350e9, 4)


def test_net_debt_to_ebitda():
    r = calculate_ratios(FUNDAMENTALS)
    assert r["leverage"]["net_debt_to_ebitda"] == round(55e9 / 130e9, 4)


def test_equity_multiplier():
    r = calculate_ratios(FUNDAMENTALS)
    assert r["leverage"]["equity_multiplier"] == round(350e9 / 60e9, 4)


# ── liquidity correctness ─────────────────────────────────────────────────────

def test_current_and_quick_ratio_passed_through():
    r = calculate_ratios(FUNDAMENTALS)["liquidity"]
    assert r["current_ratio"] == 1.15
    assert r["quick_ratio"] == 0.90


def test_working_capital_passed_through():
    r = calculate_ratios(FUNDAMENTALS)["liquidity"]
    assert r["working_capital"] == 10_000_000_000


def test_cash_ratio_derived():
    # current_liabilities ≈ working_capital / (CR - 1) = 10B / 0.15 ≈ 66.67B
    # cash_ratio = 65B / 66.67B
    r = calculate_ratios(FUNDAMENTALS)["liquidity"]
    assert r["cash_ratio"] is not None
    assert 0 < r["cash_ratio"] < 2     # sanity range


# ── valuation derived ─────────────────────────────────────────────────────────

def test_earnings_yield():
    r = calculate_ratios(FUNDAMENTALS)["valuation_derived"]
    assert r["earnings_yield"] == round(1 / 29.5, 4)


def test_fcf_yield_requires_market_data():
    r_no_md = calculate_ratios(FUNDAMENTALS)["valuation_derived"]
    assert r_no_md["fcf_yield"] is None

    r_with_md = calculate_ratios(FUNDAMENTALS, MARKET_DATA)["valuation_derived"]
    assert r_with_md["fcf_yield"] == round(109e9 / 3_000e9, 4)


def test_fcf_per_share():
    r = calculate_ratios(FUNDAMENTALS, MARKET_DATA)["valuation_derived"]
    expected = round(109e9 / 15.4e9, 4)
    assert r["fcf_per_share"] == expected


def test_price_to_fcf():
    r = calculate_ratios(FUNDAMENTALS, MARKET_DATA)["valuation_derived"]
    expected = round(3_000e9 / 109e9, 4)
    assert r["price_to_fcf"] == expected


# ── Altman Z-score ────────────────────────────────────────────────────────────

def test_altman_z_returns_none_without_market_data():
    z = calculate_ratios(FUNDAMENTALS)["risk"]["altman_z_score"]
    assert z["score"] is None
    assert "market_cap" in z["note"]


def test_altman_z_computed_with_market_data():
    z = calculate_ratios(FUNDAMENTALS, MARKET_DATA)["risk"]["altman_z_score"]
    assert z["score"] is not None
    assert isinstance(z["score"], float)
    assert z["zone"] in {"safe", "grey", "distress"}


def test_altman_z_components():
    z = calculate_ratios(FUNDAMENTALS, MARKET_DATA)["risk"]["altman_z_score"]
    # X1 = 10B / 350B
    assert z["x1_wc_to_assets"] == round(10e9 / 350e9, 6)
    # X2 always 0 (retained earnings not available)
    assert z["x2_re_to_assets"] == 0.0
    # X3 = 120B / 350B
    assert z["x3_ebit_to_assets"] == round(120e9 / 350e9, 6)
    # X4 = 3000B / 290B
    assert z["x4_mve_to_liabilities"] == round(3_000e9 / 290e9, 6)
    # X5 = 400B / 350B
    assert z["x5_revenue_to_assets"] == round(400e9 / 350e9, 6)


def test_altman_z_formula():
    z = calculate_ratios(FUNDAMENTALS, MARKET_DATA)["risk"]["altman_z_score"]
    x1 = 10e9 / 350e9
    x2 = 0.0
    x3 = 120e9 / 350e9
    x4 = 3_000e9 / 290e9
    x5 = 400e9 / 350e9
    expected = round(1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5, 4)
    assert z["score"] == expected


def test_altman_z_zone_safe():
    # AAPL-like numbers → large X4 from high market cap → should be safe
    z = calculate_ratios(FUNDAMENTALS, MARKET_DATA)["risk"]["altman_z_score"]
    assert z["zone"] == "safe"


def test_altman_z_zone_distress():
    # Company with very low market cap and high debt → distress
    distressed = {
        **FUNDAMENTALS,
        "income_statement": {
            **FUNDAMENTALS["income_statement"],
            "revenue_ttm": 500_000_000,
            "operating_income_ttm": -50_000_000,
        },
        "balance_sheet": {
            **FUNDAMENTALS["balance_sheet"],
            "total_assets": 800_000_000,
            "total_liabilities": 750_000_000,
            "working_capital": -100_000_000,
        },
    }
    distressed_market = {"market_cap": 50_000_000}
    z = calculate_ratios(distressed, distressed_market)["risk"]["altman_z_score"]
    assert z["zone"] == "distress"


# ── edge / guard cases ────────────────────────────────────────────────────────

def test_zero_equity_returns_none_for_roe():
    f = {**FUNDAMENTALS, "balance_sheet": {**FUNDAMENTALS["balance_sheet"], "total_equity": 0}}
    r = calculate_ratios(f)
    assert r["profitability"]["roe"] is None


def test_none_inputs_return_none_not_crash():
    empty = {
        "income_statement": {},
        "balance_sheet": {},
        "cash_flow": {},
        "valuation": {},
    }
    r = calculate_ratios(empty)
    assert r["profitability"]["roe"] is None
    assert r["leverage"]["debt_to_equity"] is None
    assert r["liquidity"]["current_ratio"] is None
    assert r["valuation_derived"]["earnings_yield"] is None


def test_no_market_data_param_defaults_gracefully():
    r = calculate_ratios(FUNDAMENTALS)   # no market_data arg
    assert r["valuation_derived"]["fcf_yield"] is None
    assert r["risk"]["altman_z_score"]["score"] is None
