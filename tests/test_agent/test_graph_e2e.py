"""End-to-end graph test with all nodes mocked."""
from unittest.mock import patch, MagicMock
import pytest


@patch("src.agent.nodes.critic.client")
@patch("src.agent.nodes.synthesizer.client")
@patch("src.agent.nodes.planner.client")
@patch("src.agent.nodes.tool_executor.get_market_data")
@patch("src.agent.nodes.tool_executor.get_fundamentals")
def test_graph_happy_path(mock_fundamentals, mock_market, mock_planner_c, mock_synth_c, mock_critic_c):
    # Planner: first call returns tools, second call returns empty (all gathered)
    planner_responses = [
        MagicMock(content=[MagicMock(text='{"tools": ["market_data", "fundamentals"], "reasoning": "need data"}')]),
        MagicMock(content=[MagicMock(text='{"tools": [], "reasoning": "done"}')]),
    ]
    mock_planner_c.messages.create.side_effect = planner_responses

    mock_market.return_value = {"current_price": 150, "company_name": "Apple Inc."}
    mock_fundamentals.return_value = {"income_statement": {}, "balance_sheet": {}, "valuation": {}}

    mock_synth_c.messages.create.return_value = MagicMock(
        content=[MagicMock(text="# Investment Research: AAPL\n\nFull memo here.")]
    )
    mock_critic_c.messages.create.return_value = MagicMock(
        content=[MagicMock(text='{"score": 0.85, "critique": "Looks good.", "gaps": []}')]
    )

    from src.agent.graph import build_graph
    graph = build_graph()

    result = graph.invoke({
        "ticker": "AAPL",
        "query": None,
        "messages": [],
        "plan": None,
        "findings": {},
        "memo": None,
        "critique": None,
        "critique_score": None,
        "iteration": 0,
        "errors": [],
        "status": "planning",
    })

    assert result["memo"] is not None
    assert result["critique_score"] >= 0.7
    assert result["status"] == "complete"
