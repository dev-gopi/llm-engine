"""Persistent SQLite-backed media generation job state."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TERMINAL = {"succeeded", "failed", "cancelled"}
_VALID = {"queued", "running", *_TERMINAL}


@dataclass(frozen=True)
class MediaJob:
    id: str
    kind: str
    status: str
    progress: float
    created_at: float
    updated_at: float
    request: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    cancel_requested: bool


class MediaJobStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init()

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS media_jobs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    idempotency_key TEXT
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(media_jobs)").fetchall()}
            if "idempotency_key" not in columns:
                conn.execute("ALTER TABLE media_jobs ADD COLUMN idempotency_key TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_media_jobs_updated ON media_jobs(updated_at)")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_media_jobs_idempotency ON media_jobs(idempotency_key) WHERE idempotency_key IS NOT NULL")
            # A process restart cannot safely resume an in-flight diffusion call.
            now = time.time()
            conn.execute(
                "UPDATE media_jobs SET status='failed', error='generation interrupted by server restart before completion', updated_at=? WHERE status IN ('running','queued')",
                (now,),
            )

    def create(self, kind: str, request: dict[str, Any], *, idempotency_key: str | None = None) -> MediaJob:
        job, _ = self.create_or_get(kind, request, idempotency_key=idempotency_key)
        return job

    def create_or_get(self, kind: str, request: dict[str, Any], *, idempotency_key: str | None = None) -> tuple[MediaJob, bool]:
        if kind not in {"audio", "video"}:
            raise ValueError("kind must be audio or video")
        if idempotency_key is not None:
            idempotency_key = idempotency_key.strip()
            if not idempotency_key or len(idempotency_key) > 200:
                raise ValueError("idempotency key must contain 1 to 200 characters")
        now = time.time()
        job_id = f"media_{uuid.uuid4().hex}"
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if idempotency_key is not None:
                    row = conn.execute("SELECT id FROM media_jobs WHERE idempotency_key=?", (idempotency_key,)).fetchone()
                    if row is not None:
                        conn.execute("COMMIT")
                        return self.get(row["id"]), False
                conn.execute(
                    "INSERT INTO media_jobs (id,kind,status,progress,created_at,updated_at,request_json,result_json,error,cancel_requested,idempotency_key) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (job_id, kind, "queued", 0.0, now, now, json.dumps(request), None, None, 0, idempotency_key),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return self.get(job_id), True

    def get_by_idempotency(self, key: str) -> MediaJob | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT id FROM media_jobs WHERE idempotency_key=?", (key,)).fetchone()
        return None if row is None else self.get(row["id"])

    def get(self, job_id: str) -> MediaJob:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM media_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return MediaJob(
            id=row["id"], kind=row["kind"], status=row["status"], progress=float(row["progress"]),
            created_at=float(row["created_at"]), updated_at=float(row["updated_at"]),
            request=json.loads(row["request_json"]),
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=row["error"], cancel_requested=bool(row["cancel_requested"]),
        )

    def update(self, job_id: str, *, status: str | None = None, progress: float | None = None,
               result: dict[str, Any] | None = None, error: str | None = None) -> MediaJob:
        fields, values = [], []
        if status is not None:
            if status not in _VALID:
                raise ValueError(f"invalid job status: {status}")
            fields += ["status=?"]; values += [status]
        if progress is not None:
            fields += ["progress=?"]; values += [max(0.0, min(1.0, float(progress)))]
        if result is not None:
            fields += ["result_json=?"]; values += [json.dumps(result, ensure_ascii=False)]
        if error is not None:
            fields += ["error=?"]; values += [str(error)[:8192]]
        fields += ["updated_at=?"]; values += [time.time()]
        values += [job_id]
        with self._lock, self._connect() as conn:
            cur = conn.execute(f"UPDATE media_jobs SET {', '.join(fields)} WHERE id=?", values)
            if cur.rowcount == 0:
                raise KeyError(job_id)
        return self.get(job_id)

    def request_cancel(self, job_id: str) -> MediaJob:
        job = self.get(job_id)
        if job.status in _TERMINAL:
            return job
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE media_jobs SET cancel_requested=1, updated_at=? WHERE id=?", (time.time(), job_id))
        return self.get(job_id)

    def is_cancel_requested(self, job_id: str) -> bool:
        try:
            return self.get(job_id).cancel_requested
        except KeyError:
            return True

    def list(self, *, limit: int = 50) -> list[MediaJob]:
        limit = max(1, min(500, int(limit)))
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT id FROM media_jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self.get(row["id"]) for row in rows]

    def cleanup(self, *, older_than_seconds: float) -> int:
        cutoff = time.time() - max(0.0, older_than_seconds)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM media_jobs WHERE status IN ('succeeded','failed','cancelled') AND updated_at < ?", (cutoff,)
            )
            return int(cur.rowcount)
