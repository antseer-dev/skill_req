"""
E2E test: /create_task API endpoint with mocked external HTTP calls.
Uses TestClient + httpx mocking to test full request flow without real network.
"""
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
import tempfile
import os

import api


# ── Mock HTTP response for task API ─────────────────────────────────────────

MOCK_TASK_API_DATA = {
    "data": {
        "task_id": "task-from-external-456",
        "topics": [
            {
                "topic": "inflation",
                "state": {"velocity": 0.1, "sentiment": 55},
                "posts": [
                    {
                        "id": "post_1",
                        "post_title": "DeFi yields",
                        "post_description": "DeFi yields are impossible to track manually",
                        "post_type": "tweet",
                        "creator_id": "creator_1",
                        "creator_followers": 1000,
                        "interactions_24h": 50,
                        "post_sentiment": 2.0,
                        "post_created": 1705312800,
                    },
                    {
                        "id": "post_2",
                        "post_title": "Portfolio tracking",
                        "post_description": "I have wallets on ETH SOL BSC no single tool shows my total portfolio",
                        "post_type": "reddit-post",
                        "creator_id": "creator_2",
                        "creator_followers": 500,
                        "interactions_24h": 30,
                        "post_sentiment": 1.5,
                        "post_created": 1705312800,
                    },
                ],
            }
        ],
    }
}

MOCK_MOCK_API_DATA = {
    "data": {
        "task_id": "mock-task-789",
        "topics": MOCK_TASK_API_DATA["data"]["topics"],
    }
}

MOCK_EMPTY_API_DATA = {"data": {"task_id": "task-abc", "topics": []}}


# ── Helpers ────────────────────────────────────────────────────────────────

class MockAsyncClient:
    def __init__(self, responses):
        self._responses = responses
        self._calls = []
        self._idx = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def get(self, url, **kwargs):
        self._calls.append({"url": url, "kwargs": kwargs})
        resp = MockResponse(self._responses[self._idx])
        self._idx += 1
        return resp


class MockResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._json


def make_temp_db():
    """Create a fresh temp DB file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


def init_db(db_path):
    """Create the tasks table in the given DB path."""
    import sqlite3
    with sqlite3.connect(db_path) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id           TEXT PRIMARY KEY,
                status       TEXT NOT NULL DEFAULT 'pending',
                created_at   TEXT NOT NULL,
                completed_at TEXT,
                spec_count   INTEGER,
                error        TEXT,
                specs_path   TEXT,
                report_path  TEXT
            )
        """)


@pytest.fixture
def clean_db(tmp_path):
    """Per-test isolated temp DB, patched into api module."""
    db = str(tmp_path / "tasks.db")
    init_db(db)
    original = api.DB
    api.DB = db
    yield db
    api.DB = original
    if os.path.exists(db):
        os.unlink(db)


# ── Tests ────────────────────────────────────────────────────────────────

def test_create_task_with_explicit_task_id(clean_db):
    """POST /create_task with explicit task_id returns that id."""
    mock_client = MockAsyncClient([MOCK_TASK_API_DATA])

    with patch("clients.task_client.httpx.AsyncClient", return_value=mock_client):
        with patch("api.pipeline") as mock_pipeline:
            mock_pipeline.run = AsyncMock(return_value=[])
            with TestClient(api.app) as client:
                resp = client.post("/create_task", json={"task_id": "my-task-123"})

    assert resp.status_code == 202
    data = resp.json()
    assert data["task_id"] == "my-task-123"
    assert data["status"] == "pending"


def test_create_task_main_api_falls_back_to_mock_on_error(clean_db):
    """When main API raises exception, falls back to mock API."""
    class FailingMockClient(MockAsyncClient):
        async def get(self, url, **kwargs):
            self._calls.append({"url": url, "kwargs": kwargs})
            # First call fails → fallback to mock
            raise Exception("connection refused")

    mock_client = FailingMockClient([MOCK_MOCK_API_DATA])

    with patch("clients.task_client.httpx.AsyncClient", return_value=mock_client):
        with patch("api.pipeline") as mock_pipeline:
            mock_pipeline.run = AsyncMock(return_value=[])
            with TestClient(api.app) as client:
                resp = client.post("/create_task", json={"task_id": "task-abc"})

    assert resp.status_code == 202
    # Falls back to mock API
    assert mock_client._calls[0]["url"].endswith("/api/v1/tasks")
    assert mock_client._calls[1]["url"].endswith("/api/v1/mock/tasks")


def test_create_task_without_task_id_uses_mock_api(clean_db):
    """When task_id is None, resolves from mock API."""
    mock_client = MockAsyncClient([MOCK_MOCK_API_DATA])

    with patch("clients.task_client.httpx.AsyncClient", return_value=mock_client):
        with patch("api.pipeline") as mock_pipeline:
            mock_pipeline.run = AsyncMock(return_value=[])
            with TestClient(api.app) as client:
                resp = client.post("/create_task", json={})

    assert resp.status_code == 202
    data = resp.json()
    assert data["task_id"] == "mock-task-789"


def test_create_task_no_body_rejected(clean_db):
    """POST /create_task with no body returns 422 (FastAPI validation)."""
    with TestClient(api.app) as client:
        resp = client.post("/create_task")
    assert resp.status_code == 422


def test_list_tasks_empty(clean_db):
    """GET /tasks returns empty list when no tasks exist."""
    with TestClient(api.app) as client:
        resp = client.get("/tasks")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_task_not_found(clean_db):
    """GET /tasks/{id} returns 404 for unknown id."""
    with TestClient(api.app) as client:
        resp = client.get("/tasks/nonexistent-id")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "task not found"


def test_task_flow_completed(clean_db):
    """Full flow: create_task → get_task shows completed with spec_count."""
    mock_client = MockAsyncClient([MOCK_MOCK_API_DATA])

    mock_spec = {
        "skill_name": "defi-yield-scout",
        "trigger_description": "Find and compare DeFi yields",
        "expected_output_format": "Protocol comparison table",
        "example_prompts": ["find best yield for 10k USDC"],
        "suggested_approach": "Query DeFiLlama",
        "need_name": "DeFi Yield Comparison",
        "need_type": "workflow_friction",
        "need_description": "Users manually compare yields",
        "importance": 8.5,
        "satisfaction": 3.0,
        "has_mature_solution": False,
    }

    async def mock_run(run_id=None):
        return [type("SkillSpec", (), {**mock_spec, "opportunity": type("Opp", (), {"final_score": 7.5})()})()]

    with patch("clients.task_client.httpx.AsyncClient", return_value=mock_client):
        with patch("api.pipeline") as mock_pipeline:
            mock_pipeline.run = mock_run
            with TestClient(api.app) as client:
                create_resp = client.post("/create_task", json={"task_id": "e2e-task-001"})
                assert create_resp.status_code == 202

                # Wait for background task to complete
                import time
                for _ in range(20):
                    task = client.get("/tasks/e2e-task-001").json()
                    if task["status"] in ("completed", "failed"):
                        break
                    time.sleep(0.1)

                assert task["status"] == "completed"
                assert task["spec_count"] == 1
