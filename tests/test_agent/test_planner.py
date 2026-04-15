"""Unit tests for planner node."""
from unittest.mock import patch, MagicMock
import json
import pytest


@patch("src.agent.nodes.planner.client")
def test_planner_returns_plan(mock_client):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text='{"tools": ["market_data", "fundamentals"], "reasoning": "test"}')]
    mock_client.messages.create.return_value = mock_msg

    from src.agent.nodes.planner import planner_node
    state = {
        "ticker": "AAPL",
        "query": None,
        "findings": {},
        "critique": None,
        "iteration": 0,
        "messages": [],
        "plan": None,
        "memo": None,
        "critique_score": None,
        "errors": [],
        "status": "planning",
    }
    result = planner_node(state)
    assert "plan" in result
    assert "market_data" in result["plan"]


@patch("src.agent.nodes.planner.client")
def test_planner_empty_plan_when_complete(mock_client):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text='{"tools": [], "reasoning": "all done"}')]
    mock_client.messages.create.return_value = mock_msg

    from src.agent.nodes.planner import planner_node
    state = {
        "ticker": "AAPL",
        "query": None,
        "findings": {
            "market_data": {"current_price": 150},
            "fundamentals": {},
            "ratios": {},
            "technical_signals": {},
            "sec_filings": {},
            "sec_rag_context": "text",
            "news_sentiment": {},
            "macro_context": {},
        },
        "critique": None,
        "iteration": 1,
        "messages": [],
        "plan": None,
        "memo": None,
        "critique_score": None,
        "errors": [],
        "status": "planning",
    }
    result = planner_node(state)
    assert result["plan"] == []
