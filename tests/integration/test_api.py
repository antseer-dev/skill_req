"""
E2E integration test for the FastAPI endpoints.
Runs the real pipeline (no mocks) and verifies task status + output files.

Run:
    pytest tests/integration/test_api.py -v -s
"""
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api

need_openrouter = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_KEY", ""),
    reason="OPENROUTER_KEY not set",
)


@need_openrouter
def test_task_lifecycle_e2e(tmp_path):
    """
    E2E：真实 pipeline，不 mock。
    POST /create_task → GET /tasks → GET /tasks/{id} → 验证 completed + 输出文件存在。
    """
    db_path = str(tmp_path / "tasks.db")

    # patch 必须在 TestClient 之前生效，lifespan 里 _init_db 才会在临时 DB 建表
    with __import__("unittest.mock", fromlist=["patch"]).patch.object(api, "DB", db_path):
        with TestClient(api.app) as client:

            # 1) POST /create_task
            resp = client.post("/create_task")
            assert resp.status_code == 202
            task_id = resp.json()["task_id"]
            print(f"\n  task_id: {task_id}")

            # TestClient 同步执行 background task，到这里 pipeline 已跑完

            # 2) GET /tasks/{id} — 验证状态
            detail = client.get(f"/tasks/{task_id}").json()
            print(f"  status: {detail['status']}")
            print(f"  spec_count: {detail['spec_count']}")
            print(f"  error: {detail.get('error')}")

            assert detail["status"] == "completed", (
                f"Expected completed, got {detail['status']}. Error: {detail.get('error')}"
            )
            assert detail["spec_count"] is not None
            assert detail["spec_count"] > 0
            assert detail["completed_at"] is not None
            assert detail["error"] is None

            # 3) 验证输出文件真实存在
            specs_path = detail["specs_path"]
            report_path = detail["report_path"]
            print(f"  specs_path: {specs_path}")
            print(f"  report_path: {report_path}")

            assert Path(specs_path).exists(), f"specs file not found: {specs_path}"
            assert Path(report_path).exists(), f"report file not found: {report_path}"
            assert Path(specs_path).stat().st_size > 0
            assert Path(report_path).stat().st_size > 0

            # 4) GET /tasks — 列表中能看到
            tasks = client.get("/tasks").json()
            assert len(tasks) == 1
            assert tasks[0]["id"] == task_id
            assert tasks[0]["status"] == "completed"

            # 5) GET /tasks/{不存在} — 404
            assert client.get("/tasks/nonexistent").status_code == 404

            print(f"\n  E2E passed: {detail['spec_count']} specs generated")
