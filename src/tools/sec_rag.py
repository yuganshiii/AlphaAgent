"""
SEC RAG tool — download 10-K, chunk with section-awareness, embed with
all-MiniLM-L6-v2 via SentenceTransformer, persist in ChromaDB.

Public API
----------
index_filing(filing_url, ticker) -> collection_name
query_filing(query, ticker, top_k=5) -> str
"""
import re
import requests
import chromadb
from chromadb.utils import embedding_functions

from src.config import settings

# ---------------------------------------------------------------------------
# Embedding function (module-level singleton)
# ---------------------------------------------------------------------------
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=settings.EMBEDDING_MODEL
)

# ---------------------------------------------------------------------------
# Known 10-K section headers (Item number → canonical label).
# Ordered so that two-digit items (e.g. 9A) don't match before their
# single-digit counterparts.
# ---------------------------------------------------------------------------
_SECTION_MAP = [
    (re.compile(r"\bItem\s+1A\b", re.IGNORECASE), "Item 1A. Risk Factors"),
    (re.compile(r"\bItem\s+1B\b", re.IGNORECASE), "Item 1B. Unresolved Staff Comments"),
    (re.compile(r"\bItem\s+1\b",  re.IGNORECASE), "Item 1. Business"),
    (re.compile(r"\bItem\s+2\b",  re.IGNORECASE), "Item 2. Properties"),
    (re.compile(r"\bItem\s+3\b",  re.IGNORECASE), "Item 3. Legal Proceedings"),
    (re.compile(r"\bItem\s+4\b",  re.IGNORECASE), "Item 4. Mine Safety Disclosures"),
    (re.compile(r"\bItem\s+5\b",  re.IGNORECASE), "Item 5. Market for Registrant"),
    (re.compile(r"\bItem\s+6\b",  re.IGNORECASE), "Item 6. Selected Financial Data"),
    (re.compile(r"\bItem\s+7A\b", re.IGNORECASE), "Item 7A. Quantitative and Qualitative Disclosures"),
    (re.compile(r"\bItem\s+7\b",  re.IGNORECASE), "Item 7. MD&A"),
    (re.compile(r"\bItem\s+8\b",  re.IGNORECASE), "Item 8. Financial Statements"),
    (re.compile(r"\bItem\s+9A\b", re.IGNORECASE), "Item 9A. Controls and Procedures"),
    (re.compile(r"\bItem\s+9\b",  re.IGNORECASE), "Item 9. Changes in Accountants"),
    (re.compile(r"\bItem\s+10\b", re.IGNORECASE), "Item 10. Directors and Officers"),
    (re.compile(r"\bItem\s+11\b", re.IGNORECASE), "Item 11. Executive Compensation"),
    (re.compile(r"\bItem\s+12\b", re.IGNORECASE), "Item 12. Security Ownership"),
    (re.compile(r"\bItem\s+13\b", re.IGNORECASE), "Item 13. Certain Relationships"),
    (re.compile(r"\bItem\s+14\b", re.IGNORECASE), "Item 14. Principal Accountant Fees"),
    (re.compile(r"\bItem\s+15\b", re.IGNORECASE), "Item 15. Exhibits"),
]


def _detect_section(text: str) -> str | None:
    """Return the canonical section label for the first Item header found in *text*."""
    earliest_pos = len(text) + 1
    earliest_label = None
    for pattern, label in _SECTION_MAP:
        m = pattern.search(text)
        if m and m.start() < earliest_pos:
            earliest_pos = m.start()
            earliest_label = label
    return earliest_label


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def _get_collection(ticker: str):
    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(
        name=f"sec_{ticker.lower()}",
        embedding_function=_ef,
    )


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_text_from_url(url: str) -> str:
    """Download a filing and return plain text.

    Handles both HTML (strip tags) and PDF (PyMuPDF) content types.
    Raises requests exceptions on network/HTTP errors.
    """
    headers = {"User-Agent": settings.SEC_USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        import fitz  # PyMuPDF — imported locally to keep it optional
        doc = fitz.open(stream=resp.content, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)

    # HTML / plain-text EDGAR filing
    text = re.sub(r"<[^>]+>", " ", resp.text)   # strip tags
    text = re.sub(r"\s+", " ", text)             # collapse whitespace
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[dict]:
    """Split *text* into overlapping word-count chunks with section labels.

    Each chunk dict has:
        text        – the chunk string
        section     – canonical section label (updated whenever a new Item
                      header is encountered)
        word_offset – index of the first word in the original word list
    """
    if not text:
        return []

    words = text.split()
    chunks: list[dict] = []
    i = 0
    current_section = "General"

    while i < len(words):
        window = words[i: i + chunk_size]
        chunk_text = " ".join(window)

        # Update section whenever this chunk contains a new Item header.
        detected = _detect_section(chunk_text)
        if detected:
            current_section = detected

        chunks.append({
            "text": chunk_text,
            "section": current_section,
            "word_offset": i,
        })
        i += chunk_size - overlap

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def index_filing(filing_url: str, ticker: str) -> str:
    """Download, chunk, embed, and store a 10-K filing in ChromaDB.

    Idempotent — skips re-indexing if the collection already has documents.

    Returns the ChromaDB collection name ``sec_<ticker_lower>``.
    """
    collection = _get_collection(ticker)
    if collection.count() > 0:
        return f"sec_{ticker.lower()}"

    text = _extract_text_from_url(filing_url)
    chunks = _chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

    ids = [f"{ticker.lower()}_{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [{"section": c["section"], "word_offset": c["word_offset"]} for c in chunks]

    # ChromaDB recommends batches ≤ 100 for large collections
    batch = 100
    for start in range(0, len(documents), batch):
        collection.add(
            ids=ids[start: start + batch],
            documents=documents[start: start + batch],
            metadatas=metadatas[start: start + batch],
        )

    return f"sec_{ticker.lower()}"


def query_filing(query: str, ticker: str, top_k: int = 5) -> str:
    """Semantic search over an indexed 10-K filing.

    Returns a formatted string with section labels, relevance scores, and
    the matching text chunks, separated by ``---``.
    Returns ``"Filing not yet indexed."`` if the collection is empty.
    """
    collection = _get_collection(ticker)
    if collection.count() == 0:
        return "Filing not yet indexed."

    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    parts = []
    for doc, meta, dist in zip(docs, metas, distances):
        relevance = round(1 - dist, 3)
        parts.append(
            f"[Section: {meta['section']} | Relevance: {relevance}]\n{doc}"
        )

    return "\n\n---\n\n".join(parts)
