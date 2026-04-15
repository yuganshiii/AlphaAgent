"""
Planner node — decides which tools to call next based on current state.
"""
import json
from openai import OpenAI

from src.agent.state import GraphState
from src.agent.prompts import PLANNER_SYSTEM, PLANNER_USER
from src.config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

TOOL_ORDER = [
    "market_data",
    "fundamentals",
    "ratio_calculator",
    "technical",
    "sec_edgar",
    "sec_rag",
    "news_sentiment",
    "macro",
]


def planner_node(state: GraphState) -> dict:
    findings = state.get("findings", {})
    gathered = [k for k, v in findings.items() if v is not None]

    user_msg = PLANNER_USER.format(
        ticker=state["ticker"],
        query=state.get("query") or "Full investment analysis",
        gathered=", ".join(gathered) if gathered else "none",
        critique=state.get("critique") or "none",
        iteration=state.get("iteration", 0),
        max_iterations=settings.MAX_ITERATIONS,
    )

    response = client.chat.completions.create(
        model=settings.MODEL_NAME,
        max_tokens=512,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    )

    raw = response.choices[0].message.content.strip()
    # Extract JSON even if wrapped in markdown code fences
    if "```" in raw:
        raw = raw.split("```")[1].lstrip("json").strip()

    try:
        parsed = json.loads(raw)
        plan = parsed.get("tools", [])
    except json.JSONDecodeError:
        plan = []

    return {
        "plan": plan,
        "status": "researching",
        "iteration": state.get("iteration", 0) + 1,
    }
