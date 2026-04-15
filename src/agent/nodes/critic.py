"""
Critic node — reviews the investment memo and assigns a quality score.
"""
import json
from openai import OpenAI

from src.agent.state import GraphState
from src.agent.prompts import CRITIC_SYSTEM
from src.config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)


def critic_node(state: GraphState) -> dict:
    memo = state.get("memo", "")

    response = client.chat.completions.create(
        model=settings.MODEL_NAME,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": f"Review this investment memo:\n\n{memo}"},
        ],
    )

    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1].lstrip("json").strip()

    try:
        parsed = json.loads(raw)
        score = float(parsed.get("score", 0.5))
        critique = parsed.get("critique", "")
    except (json.JSONDecodeError, ValueError):
        score = 0.5
        critique = raw

    status = "complete" if score >= settings.CRITIQUE_THRESHOLD else "planning"

    return {
        "critique": critique,
        "critique_score": score,
        "status": status,
    }
