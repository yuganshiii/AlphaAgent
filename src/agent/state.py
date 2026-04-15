from typing import TypedDict, Annotated, Optional
from operator import add


class ResearchFindings(TypedDict, total=False):
    market_data: Optional[dict]
    fundamentals: Optional[dict]
    ratios: Optional[dict]
    technical_signals: Optional[dict]
    sec_filings: Optional[dict]
    sec_rag_context: Optional[str]
    news_sentiment: Optional[dict]
    macro_context: Optional[dict]


class GraphState(TypedDict):
    ticker: str
    query: Optional[str]
    messages: Annotated[list, add]
    plan: Optional[list[str]]
    findings: ResearchFindings
    memo: Optional[str]
    critique: Optional[str]
    critique_score: Optional[float]
    iteration: int
    errors: Annotated[list[str], add]
    status: str  # "planning" | "researching" | "synthesizing" | "critiquing" | "complete"
