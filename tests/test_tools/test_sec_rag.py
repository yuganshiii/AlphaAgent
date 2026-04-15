"""Unit tests for SEC RAG tool."""
import pytest


def test_query_filing_not_indexed(tmp_path, monkeypatch):
    monkeypatch.setattr("src.tools.sec_rag.settings.CHROMA_PERSIST_DIR", str(tmp_path))
    from src.tools.sec_rag import query_filing
    result = query_filing("risk factors", "AAPL")
    assert result == "Filing not yet indexed."
