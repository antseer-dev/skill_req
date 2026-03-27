"""
Skill Req API

POST /create_task   触发 pipeline，返回 task_id
GET  /tasks         查询所有任务列表
GET  /tasks/{id}    查询单个任务；completed 时含 specs_path / report_path

Run: uvicorn api:app --reload
"""
import logging
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
import pipeline
import config

DB = "storage/tasks.db"

# ── Logging setup (mirrors pipeline.py's __main__ block) ───────────────────────
Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
log_path = f"{config.OUTPUT_DIR}/pipeline.log"
fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s", datefmt="%H:%M:%S")

file_handler = logging.FileHandler(log_path, encoding="utf-8")
file_handler.setFormatter(fmt)
console_handler = logging.StreamHandler()
console_handler.setFormatter(fmt)

logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


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


async def _run(task_id: str, pipeline_run_id: str | None, pipeline_topic: str | None):
    with _conn() as c:
        c.execute("UPDATE tasks SET status='running' WHERE id=?", (task_id,))

    try:
        specs = await pipeline.run(run_id=pipeline_run_id, topic=pipeline_topic)
        specs_path = str(Path(pipeline.config.OUTPUT_DIR).resolve() / f"skill_specs_{task_id}.json")
        report_path = str(Path(pipeline.config.OUTPUT_DIR).resolve() / f"skill_req_report_{task_id}.md")
        with _conn() as c:
            c.execute(
                "UPDATE tasks SET status='completed', completed_at=?, spec_count=?, specs_path=?, report_path=? WHERE id=?",
                (_now(), len(specs), specs_path, report_path, task_id),
            )
    except Exception as e:
        err_msg = f"{e.__class__.__name__}: {e}"
        with _conn() as c:
            c.execute("UPDATE tasks SET status='failed', completed_at=?, error=? WHERE id=?",
                      (_now(), err_msg, task_id))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/create_task", status_code=202)
async def create_task(body: dict | None, background_tasks: BackgroundTasks):
    topic = body.get("topic") if body else None
    task_id = body.get("task_id") if body else None

    if topic:
        # Topic mode: fetch single topic from mock topic API
        display_id = topic
        pipeline_run_id = topic
        pipeline_topic = topic
    elif task_id:
        # Task mode: use main API
        display_id = task_id
        pipeline_run_id = task_id
        pipeline_topic = None
    else:
        # Fallback: fetch mock all-topics API to resolve task_id
        pipeline_run_id = None
        pipeline_topic = None
        try:
            from clients.task_client import TaskClient

            client = TaskClient()
            posts, _ = await client.get_data()
            display_id = (client._task_id or "").strip() or None
        except Exception:
            display_id = None

        if not display_id:
            display_id = str(uuid.uuid4())

    with _conn() as c:
        c.execute("INSERT INTO tasks (id, created_at) VALUES (?, ?)", (display_id, _now()))

    background_tasks.add_task(_run, display_id, pipeline_run_id, pipeline_topic)
    return {"task_id": display_id, "status": "pending"}


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
