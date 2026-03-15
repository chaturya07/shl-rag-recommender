"""
pytest test suite for the SHL RAG Recommendation API.
Run: pytest tests/ -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_stats():
    res = client.get("/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["total_assessments"] == 377
    assert "supported_types" in data


def test_recommend_basic():
    res = client.post("/recommend", json={
        "query": "Senior data scientist with machine learning and Python expertise",
        "k": 5
    })
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) <= 5
    assert len(data) > 0
    assert "Assessment Name" in data[0]
    assert "URL" in data[0]
    assert "score" in data[0]


def test_recommend_returns_list():
    res = client.post("/recommend", json={
        "query": "HR manager responsible for talent acquisition and employee development",
        "k": 10
    })
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_recommend_with_type_filter():
    res = client.post("/recommend", json={
        "query": "Software engineer with problem solving skills",
        "k": 10,
        "assessment_type": "Cognitive"
    })
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)


def test_recommend_with_session():
    res = client.post("/recommend", json={
        "query": "Sales manager with leadership and negotiation skills",
        "k": 5,
        "session_id": "test_session_123"
    })
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_search_history():
    session_id = "test_history_session"

    # Make a search first
    client.post("/recommend", json={
        "query": "Customer service representative with communication skills",
        "k": 5,
        "session_id": session_id
    })

    # Get history
    res = client.get(f"/history/{session_id}")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_clear_history():
    session_id = "test_clear_session"

    # Make a search
    client.post("/recommend", json={
        "query": "Project manager with agile experience",
        "k": 5,
        "session_id": session_id
    })

    # Clear history
    res = client.delete(f"/history/{session_id}")
    assert res.status_code == 200

    # Verify cleared
    res = client.get(f"/history/{session_id}")
    assert res.json() == []


def test_export_csv():
    sample_results = [
        {"Assessment Name": "Test Assessment", "URL": "https://example.com",
         "score": 0.9, "assessment_type": "Cognitive"}
    ]
    res = client.post("/export", json={
        "results": sample_results,
        "query": "test query",
        "format": "csv"
    })
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]


def test_export_json():
    sample_results = [
        {"Assessment Name": "Test Assessment", "URL": "https://example.com",
         "score": 0.9, "assessment_type": "Cognitive"}
    ]
    res = client.post("/export", json={
        "results": sample_results,
        "query": "test query",
        "format": "json"
    })
    assert res.status_code == 200
    assert "application/json" in res.headers["content-type"]
