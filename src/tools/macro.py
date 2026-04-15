"""
Macro data tool — FRED API.
"""
import requests

from src.config import settings

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "gdp": "GDP",
    "unemployment": "UNRATE",
    "yield_curve_spread": "T10Y2Y",
    "vix": "VIXCLS",
}


def _fetch_series(series_id: str, limit: int = 3) -> list[float]:
    if not settings.FRED_API_KEY:
        return []
    params = {
        "series_id": series_id,
        "api_key": settings.FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }
    resp = requests.get(FRED_BASE, params=params, timeout=10)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    values = []
    for o in obs:
        try:
            values.append(float(o["value"]))
        except (ValueError, KeyError):
            pass
    return values


def _trend(values: list[float]) -> str:
    if len(values) < 2:
        return "unknown"
    return "rising" if values[0] > values[-1] else "falling" if values[0] < values[-1] else "stable"


def get_macro_context() -> dict:
    def _get(key: str):
        vals = _fetch_series(SERIES[key])
        latest = vals[0] if vals else None
        return {"value": latest, "trend": _trend(vals)}

    fed = _get("fed_funds_rate")
    cpi = _get("cpi")
    gdp = _get("gdp")
    unemp = _get("unemployment")
    yc = _get("yield_curve_spread")
    vix_data = _get("vix")
    vix_val = vix_data["value"]
    vix_level = (
        "extreme" if vix_val and vix_val > 40
        else "high" if vix_val and vix_val > 25
        else "moderate" if vix_val and vix_val > 15
        else "low" if vix_val
        else "unknown"
    )

    inverted = yc["value"] is not None and yc["value"] < 0

    summary_parts = []
    if fed["value"] is not None:
        summary_parts.append(f"Fed funds rate at {fed['value']}% and {fed['trend']}.")
    if cpi["value"] is not None:
        summary_parts.append(f"CPI index at {cpi['value']} ({cpi['trend']}).")
    if inverted:
        summary_parts.append("Yield curve is inverted, historically a recession signal.")
    if vix_val is not None:
        summary_parts.append(f"VIX at {vix_val} ({vix_level} volatility).")

    return {
        "fed_funds_rate": fed,
        "cpi_yoy": cpi,
        "gdp_growth": gdp,
        "unemployment": unemp,
        "yield_curve_spread": {**yc, "inverted": inverted},
        "vix": {**vix_data, "level": vix_level},
        "macro_summary": " ".join(summary_parts) or "Macro data unavailable.",
    }
