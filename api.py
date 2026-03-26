"""
Skill Req API

POST /create_task   触发 pipeline，返回 task_id
GET  /tasks         查询所有任务列表
GET  /tasks/{id}    查询单个任务；completed 时含 specs_path / report_path

Run: uvicorn api:app --reload
"""
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
import pipeline

DB = "storage/tasks.db"


def _conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def _now():
    return datetime.now(timezone.utc).isoformat()


def _init_db():
    with _conn() as c:
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


async def _run(task_id: str):
    with _conn() as c:
        c.execute("UPDATE tasks SET status='running' WHERE id=?", (task_id,))
    try:
        specs = await pipeline.run(run_id=task_id)
        specs_path = str(Path(pipeline.config.OUTPUT_DIR).resolve() / f"skill_specs_{task_id}.json")
        report_path = str(Path(pipeline.config.OUTPUT_DIR).resolve() / f"skill_req_report_{task_id}.md")
        with _conn() as c:
            c.execute(
                "UPDATE tasks SET status='completed', completed_at=?, spec_count=?, specs_path=?, report_path=? WHERE id=?",
                (_now(), len(specs), specs_path, report_path, task_id),
            )
    except Exception as e:
        with _conn() as c:
            c.execute("UPDATE tasks SET status='failed', completed_at=?, error=? WHERE id=?",
                      (_now(), str(e), task_id))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/create_task", status_code=202)
async def create_task(background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    with _conn() as c:
        c.execute("INSERT INTO tasks (id, created_at) VALUES (?, ?)", (task_id, _now()))
    background_tasks.add_task(_run, task_id)
    return {"task_id": task_id, "status": "pending"}


@app.get("/tasks")
def list_tasks():
    with _conn() as c:
        rows = c.execute(
            "SELECT id, status, created_at, completed_at, spec_count, error "
            "FROM tasks ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    with _conn() as c:
        row = c.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="task not found")
    return dict(row)
