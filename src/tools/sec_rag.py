"""
SEC RAG tool — download 10-K, chunk, embed, store in ChromaDB, query.
"""
import os
import re
import requests
import chromadb
from chromadb.utils import embedding_functions

from src.config import settings

_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=settings.EMBEDDING_MODEL
)


def _get_collection(ticker: str):
    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(
        name=f"sec_{ticker.lower()}",
        embedding_function=_ef,
    )


def _extract_text_from_url(url: str) -> str:
    """Download filing and extract plain text. Tries PDF then HTML."""
    headers = {"User-Agent": settings.SEC_USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")

    if "pdf" in content_type or url.endswith(".pdf"):
        import fitz  # PyMuPDF
        doc = fitz.open(stream=resp.content, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    else:
        # Strip HTML tags for plain text filings
        text = re.sub(r"<[^>]+>", " ", resp.text)
        text = re.sub(r"\s+", " ", text)
        return text


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """Split text into overlapping chunks; detect section headers."""
    words = text.split()
    chunks = []
    i = 0
    section = "General"
    section_re = re.compile(
        r"(Item\s+\d+[A-Za-z]?\.?\s+[A-Z][A-Za-z &]+)", re.IGNORECASE
    )

    while i < len(words):
        window = words[i: i + chunk_size]
        chunk_text = " ".join(window)
        # Detect section name in chunk
        m = section_re.search(chunk_text)
        if m:
            section = m.group(1).strip()
        chunks.append({"text": chunk_text, "section": section, "word_offset": i})
        i += chunk_size - overlap

    return chunks


def index_filing(filing_url: str, ticker: str) -> str:
    collection = _get_collection(ticker)
    # Skip re-indexing if collection already has documents
    if collection.count() > 0:
        return f"sec_{ticker.lower()}"

    text = _extract_text_from_url(filing_url)
    chunks = _chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

    ids = [f"{ticker}_{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [{"section": c["section"], "word_offset": c["word_offset"]} for c in chunks]

    # Batch insert
    batch = 100
    for start in range(0, len(documents), batch):
        collection.add(
            ids=ids[start: start + batch],
            documents=documents[start: start + batch],
            metadatas=metadatas[start: start + batch],
        )

    return f"sec_{ticker.lower()}"


def query_filing(query: str, ticker: str, top_k: int = 5) -> str:
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
        parts.append(f"[Section: {meta['section']} | Relevance: {relevance}]\n{doc}")

    return "\n\n---\n\n".join(parts)
