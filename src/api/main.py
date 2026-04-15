"""
FastAPI backend for AlphaAgent.

Endpoints:
  POST /analyze              — start analysis job
  GET  /analyze/{job_id}     — poll job status + memo
  GET  /analyze/{job_id}/stream — SSE progress stream
  GET  /health               — health check
"""
import asyncio
import json
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse

from src.api.schemas import AnalyzeRequest, AnalyzeStartResponse, JobStatusResponse
from src.api.dependencies import get_agent
from src.agent.state import GraphState

app = FastAPI(title="AlphaAgent", version="0.1.0")

# In-memory job store — replace with Redis in production
_jobs: dict[str, dict] = {}


def _initial_state(ticker: str, query: str | None) -> GraphState:
    return {
        "ticker": ticker.upper(),
        "query": query,
        "messages": [],
        "plan": None,
        "findings": {},
        "memo": None,
        "critique": None,
        "critique_score": None,
        "iteration": 0,
        "errors": [],
        "status": "planning",
    }


async def _run_agent(job_id: str, ticker: str, query: str | None):
    agent = get_agent()
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["progress"].append(f"Starting analysis for {ticker}")

    try:
        state = _initial_state(ticker, query)
        result = await asyncio.to_thread(agent.invoke, state)
        _jobs[job_id]["memo"] = result.get("memo")
        _jobs[job_id]["status"] = "complete"
        _jobs[job_id]["progress"].append("Analysis complete.")
    except Exception as exc:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(exc)
        _jobs[job_id]["progress"].append(f"Error: {exc}")


@app.post("/analyze", response_model=AnalyzeStartResponse)
async def start_analysis(req: AnalyzeRequest):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "queued",
        "memo": None,
        "progress": [],
        "error": None,
    }
    asyncio.create_task(_run_agent(job_id, req.ticker, req.query))
    return AnalyzeStartResponse(job_id=job_id)


@app.get("/analyze/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        memo=job.get("memo"),
        progress=job.get("progress", []),
        error=job.get("error"),
    )


async def _sse_generator(job_id: str) -> AsyncGenerator[str, None]:
    last_idx = 0
    while True:
        job = _jobs.get(job_id)
        if not job:
            yield f"data: {json.dumps({'event': 'error', 'message': 'job not found'})}\n\n"
            return
        progress = job.get("progress", [])
        for msg in progress[last_idx:]:
            yield f"data: {json.dumps({'event': 'progress', 'message': msg})}\n\n"
            last_idx += 1
        if job["status"] in ("complete", "error"):
            yield f"data: {json.dumps({'event': job['status'], 'memo': job.get('memo')})}\n\n"
            return
        await asyncio.sleep(1)


@app.get("/analyze/{job_id}/stream")
async def stream_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return StreamingResponse(_sse_generator(job_id), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "healthy"}
