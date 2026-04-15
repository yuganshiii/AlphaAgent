"""
Synthesizer node — combines all research findings into an investment memo.
"""
import json
from openai import OpenAI

from src.agent.state import GraphState
from src.agent.prompts import SYNTHESIZER_SYSTEM
from src.config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)


def synthesizer_node(state: GraphState) -> dict:
    findings = state.get("findings", {})
    errors = state.get("errors", [])

    system = SYNTHESIZER_SYSTEM.format(
        ticker=state["ticker"],
        company_name=(findings.get("market_data") or {}).get("company_name", state["ticker"]),
    )

    user_content = f"""
Ticker: {state["ticker"]}
User query: {state.get("query") or "Full investment analysis"}

Research findings (JSON):
{json.dumps(findings, indent=2, default=str)}

Data unavailable (tool errors):
{chr(10).join(errors) if errors else "None"}

Write the full investment memo now.
"""

    response = client.chat.completions.create(
        model=settings.MODEL_NAME,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    )

    memo = response.choices[0].message.content.strip()
    return {"memo": memo, "status": "critiquing"}
