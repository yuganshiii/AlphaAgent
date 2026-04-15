"""
Tool executor node — runs the tools requested by the planner in parallel.
"""
import asyncio
from typing import Callable

from src.agent.state import GraphState
from src.tools.market_data import get_market_data
from src.tools.fundamentals import get_fundamentals
from src.tools.ratio_calculator import calculate_ratios
from src.tools.technical import get_technical_signals
from src.tools.sec_edgar import get_sec_filings
from src.tools.sec_rag import query_filing, index_filing
from src.tools.news_sentiment import get_news_sentiment
from src.tools.macro import get_macro_context

TOOL_REGISTRY: dict[str, Callable] = {
    "market_data": get_market_data,
    "fundamentals": get_fundamentals,
    "technical": get_technical_signals,
    "sec_edgar": get_sec_filings,
    "news_sentiment": get_news_sentiment,
    "macro": get_macro_context,
}


def _run_tool(name: str, state: GraphState) -> tuple[str, object]:
    ticker = state["ticker"]
    findings = state.get("findings", {})
    try:
        if name == "ratio_calculator":
            fundamentals = findings.get("fundamentals")
            if not fundamentals:
                return name, None
            result = calculate_ratios(fundamentals)
        elif name == "sec_rag":
            sec = findings.get("sec_filings", {})
            url = sec.get("latest_10k_url") if sec else None
            if not url:
                return name, "SEC filing URL not available for RAG."
            index_filing(url, ticker)
            result = query_filing("risk factors business overview management discussion", ticker)
        elif name in TOOL_REGISTRY:
            result = TOOL_REGISTRY[name](ticker)
        else:
            return name, f"Unknown tool: {name}"
    except Exception as exc:
        return name, f"ERROR: {exc}"
    return name, result


def tool_executor_node(state: GraphState) -> dict:
    plan = state.get("plan") or []
    findings = dict(state.get("findings") or {})
    errors = []

    # Run tools — could be parallelised with threads; keep sync for simplicity
    for tool_name in plan:
        key, result = _run_tool(tool_name, state)
        if isinstance(result, str) and result.startswith("ERROR:"):
            errors.append(f"{key}: {result}")
        else:
            findings_key = "sec_rag_context" if key == "sec_rag" else key
            findings[findings_key] = result

    return {
        "findings": findings,
        "plan": [],
        "errors": errors,
        "status": "synthesizing",
    }
