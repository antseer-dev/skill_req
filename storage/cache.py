import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any

import config


class Cache:
    def __init__(self, db_path: str = config.DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = db_path
        self._init()

    def _init(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key      TEXT PRIMARY KEY,
                    value    TEXT NOT NULL,
                    expires  TEXT NOT NULL
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db)

    @staticmethod
    def _make_key(namespace: str, *parts: str) -> str:
        raw = namespace + ":" + ":".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, namespace: str, *parts: str) -> Optional[Any]:
        key = self._make_key(namespace, *parts)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value, expires FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if not row:
            return None
        value, expires = row
        if datetime.fromisoformat(expires) < datetime.utcnow():
            self.delete(namespace, *parts)
            return None
        return json.loads(value)

    def set(self, namespace: str, *parts: str, value: Any, ttl_hours: int = config.CACHE_TTL_HOURS) -> None:
        key = self._make_key(namespace, *parts)
        expires = (datetime.utcnow() + timedelta(hours=ttl_hours)).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, expires) VALUES (?, ?, ?)",
                (key, json.dumps(value), expires),
            )

    def delete(self, namespace: str, *parts: str) -> None:
        key = self._make_key(namespace, *parts)
        with self._conn() as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))

    def clear_expired(self) -> None:
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM cache WHERE expires < ?",
                (datetime.utcnow().isoformat(),),
            )
