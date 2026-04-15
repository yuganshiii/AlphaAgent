"""
CLI demo — run AlphaAgent on a ticker from the terminal.

Usage:
    python scripts/demo.py AAPL
    python scripts/demo.py TSLA "What are the main risks?"
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent.graph import agent_graph


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/demo.py <TICKER> [optional query]")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    query = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"\n{'='*60}")
    print(f"  AlphaAgent — Analyzing: {ticker}")
    if query:
        print(f"  Query: {query}")
    print(f"{'='*60}\n")

    initial_state = {
        "ticker": ticker,
        "query": query,
        "messages": [],
        "plan": None,
        "findings": {},
        "memo": None,
        "critique": None,
        "critique_score": None,
        "iteration": 0,
        "errors": [],
        "status": "planning",
    }

    result = agent_graph.invoke(initial_state)

    print("\n" + "="*60)
    print("INVESTMENT MEMO")
    print("="*60 + "\n")
    print(result.get("memo", "No memo generated."))

    if result.get("errors"):
        print("\n--- Tool Errors ---")
        for err in result["errors"]:
            print(f"  • {err}")

    print(f"\nCritique score: {result.get('critique_score')}")
    print(f"Iterations: {result.get('iteration')}")


if __name__ == "__main__":
    main()
