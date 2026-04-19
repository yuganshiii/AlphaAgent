"""
Live end-to-end demo: index a 10-K filing and query it with natural-language
questions to verify the RAG pipeline works against real EDGAR data.

Usage:
    python scripts/demo_sec_rag.py [TICKER]     # default: AAPL
    python scripts/demo_sec_rag.py MSFT
"""
import sys
import os
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tools.sec_edgar import get_sec_filings
from src.tools.sec_rag import index_filing, query_filing

QUERIES = [
    "What are the main risk factors?",
    "Describe the competitive landscape.",
    "What is the company's revenue and profitability trend?",
    "How does the company generate cash flow?",
]

WIDTH = 100


def _hr(char="─"):
    print(char * WIDTH)


def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=WIDTH - indent, initial_indent=prefix, subsequent_indent=prefix)


def main():
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"

    _hr("═")
    print(f"  SEC RAG Demo — {ticker}")
    _hr("═")

    # ── Step 1: resolve 10-K URL from EDGAR ──────────────────────────────────
    print(f"\n[1] Fetching EDGAR filing metadata for {ticker}…")
    filings = get_sec_filings(ticker)
    index_url = filings.get("latest_10k_url")
    if not index_url:
        print("  ERROR: No 10-K URL found for this ticker.")
        sys.exit(1)
    print(f"  Company : {filings['company_name']}")
    print(f"  CIK     : {filings['cik']}")
    print(f"  Index   : {index_url}")

    # ── Step 2: index the filing ─────────────────────────────────────────────
    print(f"\n[2] Indexing 10-K into ChromaDB…")
    print("  (downloading and chunking the filing — this may take ~30 s on first run)")
    collection_name = index_filing(index_url, ticker)
    print(f"  Collection: {collection_name}")

    # ── Step 3: query with test questions ────────────────────────────────────
    print(f"\n[3] Querying with {len(QUERIES)} questions (top_k=3, min_relevance=0.4)…")
    for i, query in enumerate(QUERIES, 1):
        _hr()
        print(f"  Q{i}: {query}")
        _hr()
        result = query_filing(query, ticker, top_k=3, min_relevance=0.4)
        for block in result.split("\n\n---\n\n"):
            header, _, body = block.partition("\n")
            print(f"  {header}")
            # Print first 400 chars of body
            snippet = body[:400].replace("\n", " ")
            if len(body) > 400:
                snippet += " …"
            print(_wrap(snippet))
            print()

    _hr("═")
    print("  Demo complete.")
    _hr("═")


if __name__ == "__main__":
    main()
