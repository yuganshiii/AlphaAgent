from typing import Optional
from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    ticker: str
    query: Optional[str] = None


class AnalyzeStartResponse(BaseModel):
    job_id: str
    status: str = "started"


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "running" | "complete" | "error"
    memo: Optional[str] = None
    progress: list[str] = []
    error: Optional[str] = None
