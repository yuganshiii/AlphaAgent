"""
One-time script — download and index a 10-K for demo/testing.

Usage:
    python scripts/seed_rag.py AAPL
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tools.sec_edgar import get_sec_filings
from src.tools.sec_rag import index_filing


def main():
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"
    print(f"Fetching SEC filings for {ticker}…")
    filings = get_sec_filings(ticker)
    url = filings.get("latest_10k_url")
    if not url:
        print("No 10-K URL found.")
        sys.exit(1)
    print(f"10-K URL: {url}")
    print("Indexing filing into ChromaDB…")
    collection_name = index_filing(url, ticker)
    print(f"Done. Collection: {collection_name}")


if __name__ == "__main__":
    main()
