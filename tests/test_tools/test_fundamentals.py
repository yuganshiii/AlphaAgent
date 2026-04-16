"""Unit tests for fundamentals tool."""
from unittest.mock import patch, MagicMock, PropertyMock
import pandas as pd
import pytest


# ── fixtures ──────────────────────────────────────────────────────────────────

VALID_INFO = {
    "longName": "Apple Inc.",
    # income
    "totalRevenue": 400_000_000_000,
    "revenueGrowth": 0.08,
    "grossProfits": 175_000_000_000,
    "grossMargins": 0.4375,
    "operatingMargins": 0.295,
    "profitMargins": 0.255,
    "netIncomeToCommon": 102_000_000_000,
    "ebitda": 130_000_000_000,
    "trailingEps": 6.42,
    "earningsGrowth": 0.10,
    "sharesOutstanding": 15_400_000_000,
    "operatingCashflow": 120_000_000_000,
    "capitalExpenditures": 11_000_000_000,
    "freeCashflow": 109_000_000_000,
    # balance sheet
    "totalAssets": 353_000_000_000,
    "totalLiab": 290_000_000_000,
    "totalStockholderEquity": 63_000_000_000,
    "cash": 30_000_000_000,
    "shortTermInvestments": 35_000_000_000,
    "totalCash": 65_000_000_000,
    "totalDebt": 120_000_000_000,
    "bookValue": 4.10,
    "totalCurrentAssets": 140_000_000_000,
    "totalCurrentLiabilities": 130_000_000_000,
    # valuation
    "trailingPE": 29.5,
    "forwardPE": 26.0,
    "priceToBook": 45.2,
    "priceToSalesTrailing12Months": 7.5,
    "enterpriseToEbitda": 24.0,
    "enterpriseToRevenue": 7.8,
    "pegRatio": 2.9,
    "dividendYield": 0.0055,
    "payoutRatio": 0.16,
    "enterpriseValue": 3_100_000_000_000,
}


def _mock_ticker(info=None, income_stmt=None):
    mock = MagicMock()
    mock.info = info if info is not None else VALID_INFO
    mock.income_stmt = income_stmt if income_stmt is not None else pd.DataFrame()
    return mock


# ── structure tests ───────────────────────────────────────────────────────────

@patch("src.tools.fundamentals.yf.Ticker")
def test_top_level_keys(mock_cls):
    mock_cls.return_value = _mock_ticker()
    from src.tools.fundamentals import get_fundamentals
    result = get_fundamentals("AAPL")
    assert set(result.keys()) == {"income_statement", "balance_sheet", "cash_flow", "valuation"}


@patch("src.tools.fundamentals.yf.Ticker")
def test_income_statement_keys(mock_cls):
    mock_cls.return_value = _mock_ticker()
    from src.tools.fundamentals import get_fundamentals
    inc = get_fundamentals("AAPL")["income_statement"]
    expected = {
        "revenue_ttm", "revenue_growth_yoy", "gross_profit_ttm", "gross_margin",
        "operating_income_ttm", "operating_margin", "net_income_ttm", "net_margin",
        "ebitda_ttm", "eps_ttm", "eps_diluted_ttm", "eps_growth_yoy",
        "shares_outstanding", "revenue_history",
    }
    assert expected == set(inc.keys())


@patch("src.tools.fundamentals.yf.Ticker")
def test_balance_sheet_keys(mock_cls):
    mock_cls.return_value = _mock_ticker()
    from src.tools.fundamentals import get_fundamentals
    bs = get_fundamentals("AAPL")["balance_sheet"]
    expected = {
        "total_assets", "total_liabilities", "total_equity",
        "cash_and_equivalents", "short_term_investments", "total_cash",
        "total_debt", "net_debt", "working_capital", "book_value_per_share",
    }
    assert expected == set(bs.keys())


@patch("src.tools.fundamentals.yf.Ticker")
def test_cash_flow_keys(mock_cls):
    mock_cls.return_value = _mock_ticker()
    from src.tools.fundamentals import get_fundamentals
    cf = get_fundamentals("AAPL")["cash_flow"]
    assert set(cf.keys()) == {"operating_cash_flow_ttm", "capex_ttm", "free_cash_flow_ttm", "fcf_margin"}


@patch("src.tools.fundamentals.yf.Ticker")
def test_valuation_keys(mock_cls):
    mock_cls.return_value = _mock_ticker()
    from src.tools.fundamentals import get_fundamentals
    val = get_fundamentals("AAPL")["valuation"]
    expected = {
        "pe_ratio", "forward_pe", "pb_ratio", "ps_ratio",
        "ev_ebitda", "ev_revenue", "peg_ratio",
        "dividend_yield", "payout_ratio", "enterprise_value",
    }
    assert expected == set(val.keys())


# ── value correctness tests ───────────────────────────────────────────────────

@patch("src.tools.fundamentals.yf.Ticker")
def test_income_values(mock_cls):
    mock_cls.return_value = _mock_ticker()
    from src.tools.fundamentals import get_fundamentals
    inc = get_fundamentals("AAPL")["income_statement"]
    assert inc["revenue_ttm"] == 400_000_000_000
    assert inc["net_income_ttm"] == 102_000_000_000
    assert inc["ebitda_ttm"] == 130_000_000_000
    assert inc["eps_ttm"] == 6.42
    assert inc["gross_margin"] == 0.4375
    assert inc["net_margin"] == 0.255


@patch("src.tools.fundamentals.yf.Ticker")
def test_net_debt_calculated(mock_cls):
    mock_cls.return_value = _mock_ticker()
    from src.tools.fundamentals import get_fundamentals
    bs = get_fundamentals("AAPL")["balance_sheet"]
    # net_debt = total_debt - total_cash = 120B - 65B
    assert bs["net_debt"] == 120_000_000_000 - 65_000_000_000


@patch("src.tools.fundamentals.yf.Ticker")
def test_fcf_margin_calculated(mock_cls):
    mock_cls.return_value = _mock_ticker()
    from src.tools.fundamentals import get_fundamentals
    cf = get_fundamentals("AAPL")["cash_flow"]
    # capex is positive in VALID_INFO → FCF = OCF - capex
    expected_fcf = 120_000_000_000 - 11_000_000_000
    assert cf["free_cash_flow_ttm"] == expected_fcf
    assert cf["fcf_margin"] == round(expected_fcf / 400_000_000_000, 6)


@patch("src.tools.fundamentals.yf.Ticker")
def test_valuation_values(mock_cls):
    mock_cls.return_value = _mock_ticker()
    from src.tools.fundamentals import get_fundamentals
    val = get_fundamentals("AAPL")["valuation"]
    assert val["pe_ratio"] == 29.5
    assert val["forward_pe"] == 26.0
    assert val["ev_ebitda"] == 24.0
    assert val["dividend_yield"] == 0.0055


# ── revenue history from income_stmt DataFrame ────────────────────────────────

@patch("src.tools.fundamentals.yf.Ticker")
def test_revenue_history_from_dataframe(mock_cls):
    # yfinance shape: index = line-item labels, columns = fiscal year end dates
    dates = pd.to_datetime(["2024-09-30", "2023-09-30", "2022-09-30", "2021-09-30"])
    revenues = [400e9, 383e9, 394e9, 365e9]
    income_stmt = pd.DataFrame(
        [revenues],
        index=pd.Index(["Total Revenue"]),
        columns=dates,
    )
    mock_cls.return_value = _mock_ticker(income_stmt=income_stmt)
    from src.tools.fundamentals import get_fundamentals
    result = get_fundamentals("AAPL")
    history = result["income_statement"]["revenue_history"]
    assert len(history) == 4
    assert history[0]["value"] == int(365e9)   # oldest first after sort
    assert "period" in history[0]


# ── error handling ────────────────────────────────────────────────────────────

@patch("src.tools.fundamentals.yf.Ticker")
def test_invalid_ticker_raises(mock_cls):
    mock_cls.return_value = _mock_ticker(info={})
    from src.tools.fundamentals import get_fundamentals
    with pytest.raises(ValueError, match="not found"):
        get_fundamentals("INVALIDXYZ")


@patch("src.tools.fundamentals.yf.Ticker")
def test_info_exception_raises(mock_cls):
    mock = MagicMock()
    type(mock).info = PropertyMock(side_effect=Exception("timeout"))
    mock_cls.return_value = mock
    from src.tools.fundamentals import get_fundamentals
    with pytest.raises(ValueError):
        get_fundamentals("AAPL")


@patch("src.tools.fundamentals.yf.Ticker")
def test_missing_optional_fields_are_none(mock_cls):
    minimal = {"longName": "Minimal Corp", "totalRevenue": 1_000_000}
    mock_cls.return_value = _mock_ticker(info=minimal)
    from src.tools.fundamentals import get_fundamentals
    result = get_fundamentals("MINI")
    assert result["balance_sheet"]["net_debt"] is None
    assert result["cash_flow"]["fcf_margin"] is None
    assert result["valuation"]["pe_ratio"] is None


# ── FMP supplement ────────────────────────────────────────────────────────────

@patch("src.tools.fundamentals._fmp_get")
@patch("src.tools.fundamentals.yf.Ticker")
def test_fmp_fills_none_fields(mock_cls, mock_fmp, monkeypatch):
    monkeypatch.setattr("src.tools.fundamentals.settings.FMP_API_KEY", "test_key")

    # yfinance returns no EPS or EV
    sparse_info = {**VALID_INFO, "trailingEps": None, "enterpriseValue": None}
    mock_cls.return_value = _mock_ticker(info=sparse_info)

    # FMP returns data
    fmp_inc = [{"revenue": 400e9, "eps": 6.55, "epsdiluted": 6.50,
                "grossProfit": 175e9, "operatingIncome": 118e9,
                "netIncome": 102e9, "ebitda": 130e9}]
    fmp_bs  = [{"totalAssets": 353e9, "totalLiabilities": 290e9,
                "totalStockholdersEquity": 63e9,
                "cashAndShortTermInvestments": 65e9, "totalDebt": 120e9,
                "bookValuePerShare": 4.10}]
    fmp_cf  = [{"operatingCashFlow": 120e9, "capitalExpenditure": -11e9, "freeCashFlow": 109e9}]
    fmp_km  = [{"peRatioTTM": 29.5, "pbRatioTTM": 45.2,
                "priceToSalesRatioTTM": 7.5, "evToEbitdaTTM": 24.0,
                "evToSalesTTM": 7.8, "pegRatioTTM": 2.9,
                "enterpriseValueTTM": 3_100_000_000_000}]

    # _fmp_get is called for income (ttm), income (annual list), balance, cf, key-metrics
    mock_fmp.side_effect = [fmp_inc, fmp_inc, fmp_bs, fmp_cf, fmp_km]

    from src.tools.fundamentals import get_fundamentals
    result = get_fundamentals("AAPL")

    assert result["income_statement"]["eps_ttm"] == 6.55
    assert result["valuation"]["enterprise_value"] == 3_100_000_000_000


@patch("src.tools.fundamentals._fmp_get")
@patch("src.tools.fundamentals.yf.Ticker")
def test_fmp_failure_does_not_crash(mock_cls, mock_fmp, monkeypatch):
    monkeypatch.setattr("src.tools.fundamentals.settings.FMP_API_KEY", "test_key")
    mock_cls.return_value = _mock_ticker()
    mock_fmp.side_effect = Exception("FMP rate limited")

    from src.tools.fundamentals import get_fundamentals
    result = get_fundamentals("AAPL")     # should not raise
    assert result["income_statement"]["revenue_ttm"] == 400_000_000_000
