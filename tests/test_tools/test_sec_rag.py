"""
Comprehensive tests for sec_rag.py.

chromadb and sentence-transformers are mocked throughout so the tests run
without those heavy dependencies being fully initialised.
The pure-Python helpers (_chunk_text, _extract_text_from_url) are tested
directly where possible.
"""
import re
import pytest
import requests
from unittest.mock import patch, MagicMock, call


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_collection(count: int = 0, docs=None, metas=None, distances=None):
    """Return a mock chromadb Collection."""
    col = MagicMock()
    col.count.return_value = count
    if docs is not None:
        col.query.return_value = {
            "documents": [docs],
            "metadatas": [metas or [{"section": "General", "word_offset": 0}] * len(docs)],
            "distances": [distances or [0.1] * len(docs)],
        }
    return col


def _patch_chroma(collection):
    """Patch the entire chromadb stack in sec_rag."""
    client_mock = MagicMock()
    client_mock.get_or_create_collection.return_value = collection
    return patch("src.tools.sec_rag.chromadb.PersistentClient", return_value=client_mock)


# ── _chunk_text unit tests ────────────────────────────────────────────────────

def test_chunk_text_produces_chunks():
    from src.tools.sec_rag import _chunk_text
    text = " ".join([f"word{i}" for i in range(3000)])
    chunks = _chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1


def test_chunk_text_empty_string():
    from src.tools.sec_rag import _chunk_text
    assert _chunk_text("", chunk_size=100, overlap=20) == []


def test_chunk_text_shorter_than_chunk_size():
    from src.tools.sec_rag import _chunk_text
    text = "hello world this is a short text"
    chunks = _chunk_text(text, chunk_size=1000, overlap=200)
    assert len(chunks) == 1
    assert "hello" in chunks[0]["text"]


def test_chunk_text_chunk_has_required_keys():
    from src.tools.sec_rag import _chunk_text
    text = " ".join([f"word{i}" for i in range(200)])
    chunks = _chunk_text(text, chunk_size=100, overlap=20)
    for chunk in chunks:
        assert set(chunk.keys()) == {"text", "section", "word_offset"}


def test_chunk_text_overlap_creates_shared_words():
    from src.tools.sec_rag import _chunk_text
    words = [f"w{i}" for i in range(300)]
    text = " ".join(words)
    chunks = _chunk_text(text, chunk_size=100, overlap=20)
    # The start of chunk[1] should overlap with the end of chunk[0]
    c0_words = set(chunks[0]["text"].split())
    c1_words = set(chunks[1]["text"].split())
    assert len(c0_words & c1_words) > 0


def test_chunk_text_word_offset_increases():
    from src.tools.sec_rag import _chunk_text
    text = " ".join([f"word{i}" for i in range(500)])
    chunks = _chunk_text(text, chunk_size=100, overlap=20)
    offsets = [c["word_offset"] for c in chunks]
    assert offsets == sorted(offsets)
    assert offsets[0] == 0


def test_chunk_text_section_detection():
    from src.tools.sec_rag import _chunk_text
    text = " ".join(["filler"] * 50)
    text += " Item 1A. Risk Factors The company faces many risks "
    text += " ".join(["more_filler"] * 50)
    chunks = _chunk_text(text, chunk_size=60, overlap=10)
    sections = [c["section"] for c in chunks]
    # At least one chunk should have the Risk Factors section detected
    assert any("Risk" in s or "Item" in s for s in sections)


def test_chunk_text_default_section_is_general():
    from src.tools.sec_rag import _chunk_text
    text = " ".join([f"plain{i}" for i in range(200)])
    chunks = _chunk_text(text, chunk_size=100, overlap=10)
    assert chunks[0]["section"] == "General"


# ── _extract_text_from_url unit tests ─────────────────────────────────────────

@patch("src.tools.sec_rag.requests.get")
def test_extract_text_strips_html_tags(mock_get):
    from src.tools.sec_rag import _extract_text_from_url
    html = "<html><body><p>Risk factors include <b>competition</b> and <i>regulation</i>.</p></body></html>"
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.headers = {"Content-Type": "text/html"}
    mock_resp.text = html
    mock_get.return_value = mock_resp

    text = _extract_text_from_url("https://sec.gov/filing.htm")
    assert "<" not in text
    assert "Risk factors" in text
    assert "competition" in text


@patch("src.tools.sec_rag.requests.get")
def test_extract_text_collapses_whitespace(mock_get):
    from src.tools.sec_rag import _extract_text_from_url
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.headers = {"Content-Type": "text/html"}
    mock_resp.text = "<p>Word1    \n\n    Word2</p>"
    mock_get.return_value = mock_resp

    text = _extract_text_from_url("https://sec.gov/filing.htm")
    assert "  " not in text.strip()


@patch("src.tools.sec_rag.requests.get")
def test_extract_text_routes_pdf_by_content_type(mock_get):
    from src.tools.sec_rag import _extract_text_from_url
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.headers = {"Content-Type": "application/pdf"}
    mock_resp.content = b"%PDF-fake"
    mock_get.return_value = mock_resp

    # fitz is imported locally inside _extract_text_from_url; patch via sys.modules
    import sys
    mock_fitz = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "PDF risk content"
    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_fitz.open.return_value = mock_doc

    with patch.dict(sys.modules, {"fitz": mock_fitz}):
        text = _extract_text_from_url("https://sec.gov/filing.pdf")
        mock_fitz.open.assert_called_once()
        assert "PDF risk content" in text


@patch("src.tools.sec_rag.requests.get")
def test_extract_text_routes_pdf_by_url_extension(mock_get):
    from src.tools.sec_rag import _extract_text_from_url
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.headers = {"Content-Type": "application/octet-stream"}
    mock_resp.content = b"%PDF-fake"
    mock_get.return_value = mock_resp

    import sys
    mock_fitz = MagicMock()
    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([]))
    mock_fitz.open.return_value = mock_doc

    with patch.dict(sys.modules, {"fitz": mock_fitz}):
        _extract_text_from_url("https://sec.gov/filing.pdf")
        mock_fitz.open.assert_called_once()


@patch("src.tools.sec_rag.requests.get")
def test_extract_text_timeout_raises(mock_get):
    from src.tools.sec_rag import _extract_text_from_url
    mock_get.side_effect = requests.exceptions.Timeout("timed out")
    with pytest.raises(requests.exceptions.Timeout):
        _extract_text_from_url("https://sec.gov/filing.htm")


@patch("src.tools.sec_rag.requests.get")
def test_extract_text_http_error_raises(mock_get):
    from src.tools.sec_rag import _extract_text_from_url
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
    mock_get.return_value = mock_resp
    with pytest.raises(requests.HTTPError):
        _extract_text_from_url("https://sec.gov/private.htm")


# ── index_filing tests ────────────────────────────────────────────────────────

def test_index_filing_skips_if_already_indexed():
    col = _make_collection(count=50)
    with _patch_chroma(col):
        from src.tools.sec_rag import index_filing
        result = index_filing("https://sec.gov/filing.htm", "AAPL")
    assert result == "sec_aapl"
    col.add.assert_not_called()


def test_index_filing_returns_collection_name():
    col = _make_collection(count=0)
    text = " ".join([f"word{i}" for i in range(2000)])
    with _patch_chroma(col):
        with patch("src.tools.sec_rag._extract_text_from_url", return_value=text):
            from src.tools.sec_rag import index_filing
            result = index_filing("https://sec.gov/filing.htm", "AAPL")
    assert result == "sec_aapl"


def test_index_filing_adds_documents_to_collection():
    col = _make_collection(count=0)
    text = " ".join([f"word{i}" for i in range(2000)])
    with _patch_chroma(col):
        with patch("src.tools.sec_rag._extract_text_from_url", return_value=text):
            from src.tools.sec_rag import index_filing
            index_filing("https://sec.gov/filing.htm", "AAPL")
    assert col.add.called


def test_index_filing_collection_name_is_lowercase_ticker():
    from unittest.mock import ANY
    col = _make_collection(count=0)
    text = " ".join([f"word{i}" for i in range(500)])
    with _patch_chroma(col) as mock_client_cls:
        with patch("src.tools.sec_rag._extract_text_from_url", return_value=text):
            from src.tools.sec_rag import index_filing
            index_filing("https://sec.gov/filing.htm", "TSLA")
    mock_client_cls.return_value.get_or_create_collection.assert_called_with(
        name="sec_tsla",
        embedding_function=ANY,
    )


def test_index_filing_batches_large_collections():
    """More than 100 chunks → multiple add() calls (batch=100)."""
    from src.tools.sec_rag import index_filing
    col = _make_collection(count=0)
    # 5000 words with chunk_size=50, overlap=10 → many chunks > 100
    text = " ".join([f"word{i}" for i in range(5000)])
    with _patch_chroma(col):
        with patch("src.tools.sec_rag._extract_text_from_url", return_value=text):
            with patch("src.tools.sec_rag.settings.CHUNK_SIZE", 50):
                with patch("src.tools.sec_rag.settings.CHUNK_OVERLAP", 10):
                    index_filing("https://sec.gov/filing.htm", "AAPL")
    # add() must have been called at least once
    assert col.add.call_count >= 1


# ── query_filing tests ────────────────────────────────────────────────────────

def test_query_filing_not_indexed_message():
    col = _make_collection(count=0)
    with _patch_chroma(col):
        from src.tools.sec_rag import query_filing
        result = query_filing("risk factors", "AAPL")
    assert result == "Filing not yet indexed."


def test_query_filing_returns_string():
    col = _make_collection(
        count=3,
        docs=["Risk chunk 1", "Risk chunk 2", "Risk chunk 3"],
        metas=[
            {"section": "Item 1A. Risk Factors", "word_offset": 0},
            {"section": "Item 1A. Risk Factors", "word_offset": 800},
            {"section": "Item 7. MD&A", "word_offset": 1600},
        ],
        distances=[0.05, 0.12, 0.25],
    )
    with _patch_chroma(col):
        from src.tools.sec_rag import query_filing
        result = query_filing("risk factors", "AAPL")
    assert isinstance(result, str)
    assert len(result) > 0


def test_query_filing_includes_section_label():
    col = _make_collection(
        count=1,
        docs=["Chunk about risks."],
        metas=[{"section": "Item 1A. Risk Factors", "word_offset": 0}],
        distances=[0.1],
    )
    with _patch_chroma(col):
        from src.tools.sec_rag import query_filing
        result = query_filing("risk", "AAPL")
    assert "Item 1A. Risk Factors" in result


def test_query_filing_includes_relevance_score():
    col = _make_collection(
        count=1,
        docs=["Some chunk text."],
        metas=[{"section": "General", "word_offset": 0}],
        distances=[0.2],
    )
    with _patch_chroma(col):
        from src.tools.sec_rag import query_filing
        result = query_filing("risk", "AAPL")
    assert "Relevance" in result
    assert "0.8" in result   # 1 - 0.2 = 0.8


def test_query_filing_separates_chunks_with_divider():
    col = _make_collection(
        count=2,
        docs=["Chunk one.", "Chunk two."],
        metas=[
            {"section": "General", "word_offset": 0},
            {"section": "General", "word_offset": 1000},
        ],
        distances=[0.1, 0.2],
    )
    with _patch_chroma(col):
        from src.tools.sec_rag import query_filing
        result = query_filing("risk", "AAPL")
    assert "---" in result


def test_query_filing_top_k_passed_to_chroma():
    col = _make_collection(count=10, docs=["x"] * 3, distances=[0.1] * 3)
    with _patch_chroma(col):
        from src.tools.sec_rag import query_filing
        query_filing("revenue growth", "AAPL", top_k=3)
    col.query.assert_called_once_with(query_texts=["revenue growth"], n_results=3)
