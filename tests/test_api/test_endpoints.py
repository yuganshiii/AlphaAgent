"""API endpoint tests."""
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_analyze_invalid_ticker_still_queues():
    # The API queues the job regardless; errors surface via polling
    resp = client.post("/analyze", json={"ticker": "FAKEXYZ"})
    assert resp.status_code == 200
    assert "job_id" in resp.json()


def test_analyze_poll_not_found():
    resp = client.get("/analyze/nonexistent-job-id")
    assert resp.status_code == 404


def test_analyze_and_poll():
    resp = client.post("/analyze", json={"ticker": "AAPL", "query": "What is the debt level?"})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    poll = client.get(f"/analyze/{job_id}")
    assert poll.status_code == 200
    assert poll.json()["status"] in {"queued", "running", "complete", "error"}
