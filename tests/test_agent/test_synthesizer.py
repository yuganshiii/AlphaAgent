"""Unit tests for synthesizer node."""
from unittest.mock import patch, MagicMock
import pytest


@patch("src.agent.nodes.synthesizer.client")
def test_synthesizer_returns_memo(mock_client):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text="# Investment Research: AAPL\n\n## Executive Summary\nApple is great.")]
    mock_client.messages.create.return_value = mock_msg

    from src.agent.nodes.synthesizer import synthesizer_node
    state = {
        "ticker": "AAPL",
        "query": None,
        "findings": {"market_data": {"current_price": 150, "company_name": "Apple Inc."}},
        "errors": [],
        "messages": [],
        "plan": [],
        "memo": None,
        "critique": None,
        "critique_score": None,
        "iteration": 1,
        "status": "synthesizing",
    }
    result = synthesizer_node(state)
    assert "memo" in result
    assert "AAPL" in result["memo"]
