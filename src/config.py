import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME: str = "gpt-4o-mini"

    # External APIs
    FMP_API_KEY: str = os.getenv("FMP_API_KEY", "")
    FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")
    FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")

    # SEC EDGAR — no auth, just need a User-Agent
    SEC_USER_AGENT: str = "AlphaAgent yuganshi@umich.edu"

    # RAG / ChromaDB
    CHROMA_PERSIST_DIR: str = os.path.join(os.path.dirname(__file__), "..", "data", "chroma")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Agent
    MAX_ITERATIONS: int = 3
    CRITIQUE_THRESHOLD: float = 0.7


settings = Settings()
