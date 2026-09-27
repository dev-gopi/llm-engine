"""Small persistent OpenAI-compatible platform APIs.

The goal is API/lifecycle compatibility for local deployments without forcing an
external control plane. Files and job metadata are stored in SQLite; file bytes
live under a configurable directory. Batch/fine-tuning jobs are persistent
control-plane records and can be picked up by external workers.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VectorStoreCreate(_Strict):
    name: str = Field(min_length=1, max_length=256)
    expires_after_seconds: int | None = Field(default=None, ge=60, le=31_536_000)
    metadata: dict[str, str] = Field(default_factory=dict)


class VectorStoreFileCreate(_Strict):
    file_id: str = Field(min_length=1, max_length=128)


class BatchCreate(_Strict):
    input_file_id: str = Field(min_length=1, max_length=128)
    endpoint: str = Field(pattern=r"^/v1/(responses|chat/completions|embeddings)$")
    completion_window: str = Field(default="24h", pattern=r"^24h$")
    metadata: dict[str, str] = Field(default_factory=dict)


class FineTuningJobCreate(_Strict):
    training_file: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=256)
    validation_file: str | None = Field(default=None, max_length=128)
    suffix: str | None = Field(default=None, max_length=64)
    method: dict[str, Any] | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class ImageGenerationRequest(_Strict):
    prompt: str = Field(min_length=1, max_length=4096)
    model: str | None = Field(default=None, max_length=256)
    n: int = Field(default=1, ge=1, le=10)
    size: str = Field(default="1024x1024", pattern=r"^\d{2,4}x\d{2,4}$")
    quality: str = Field(default="standard", pattern=r"^(standard|hd|draft|quality|max_quality)$")
    response_format: str = Field(default="url", pattern=r"^(url|b64_json)$")
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)


class PlatformStore:
    def __init__(self, db_path: str | Path, files_dir: str | Path) -> None:
        self.db_path = Path(db_path)
        self.files_dir = Path(files_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS files(
                    id TEXT PRIMARY KEY, tenant TEXT NOT NULL, filename TEXT NOT NULL,
                    purpose TEXT NOT NULL, bytes INTEGER NOT NULL, sha256 TEXT NOT NULL,
                    created_at INTEGER NOT NULL, path TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS files_tenant_idx ON files(tenant,created_at);
                CREATE TABLE IF NOT EXISTS vector_stores(
                    id TEXT PRIMARY KEY, tenant TEXT NOT NULL, name TEXT NOT NULL,
                    created_at INTEGER NOT NULL, expires_at INTEGER, status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS vector_store_files(
                    vector_store_id TEXT NOT NULL, file_id TEXT NOT NULL, tenant TEXT NOT NULL,
                    status TEXT NOT NULL, created_at INTEGER NOT NULL,
                    PRIMARY KEY(vector_store_id,file_id)
                );
                CREATE TABLE IF NOT EXISTS batches(
                    id TEXT PRIMARY KEY, tenant TEXT NOT NULL, input_file_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL, completion_window TEXT NOT NULL,
                    status TEXT NOT NULL, created_at INTEGER NOT NULL,
                    cancelled_at INTEGER, output_file_id TEXT, error_file_id TEXT,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS fine_tuning_jobs(
                    id TEXT PRIMARY KEY, tenant TEXT NOT NULL, training_file TEXT NOT NULL,
                    validation_file TEXT, model TEXT NOT NULL, status TEXT NOT NULL,
                    created_at INTEGER NOT NULL, finished_at INTEGER, suffix TEXT,
                    method_json TEXT, hyperparameters_json TEXT NOT NULL,
                    fine_tuned_model TEXT, error_json TEXT
                );
                CREATE TABLE IF NOT EXISTS idempotency(
                    tenant TEXT NOT NULL, route TEXT NOT NULL, idem_key TEXT NOT NULL,
                    request_hash TEXT NOT NULL, status_code INTEGER NOT NULL,
                    response_json TEXT NOT NULL, created_at INTEGER NOT NULL,
                    PRIMARY KEY(tenant,route,idem_key)
                );
                """
            )

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _now() -> int:
        return int(time.time())

    def put_file(self, tenant: str, filename: str, purpose: str, content: bytes) -> dict[str, Any]:
        if len(content) > int(os.getenv("GOPI_FILES_MAX_BYTES", str(100 * 1024 * 1024))):
            raise ValueError("file exceeds configured size limit")
        file_id = f"file-{uuid.uuid4().hex}"
        safe = Path(filename).name or "upload.bin"
        path = self.files_dir / f"{file_id}-{safe}"
        path.write_bytes(content)
        now = self._now()
        digest = hashlib.sha256(content).hexdigest()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO files VALUES(?,?,?,?,?,?,?,?)",
                (file_id, tenant, safe, purpose, len(content), digest, now, str(path)),
            )
        return {"id": file_id, "object": "file", "bytes": len(content), "created_at": now,
                "filename": safe, "purpose": purpose, "status": "processed", "sha256": digest}

    def get_file(self, tenant: str, file_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute("SELECT * FROM files WHERE id=? AND tenant=?", (file_id, tenant)).fetchone()

    def list_files(self, tenant: str, purpose: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if purpose:
                rows = conn.execute("SELECT * FROM files WHERE tenant=? AND purpose=? ORDER BY created_at DESC", (tenant, purpose)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM files WHERE tenant=? ORDER BY created_at DESC", (tenant,)).fetchall()
        return [{"id": r["id"], "object": "file", "bytes": r["bytes"], "created_at": r["created_at"],
                 "filename": r["filename"], "purpose": r["purpose"], "status": "processed", "sha256": r["sha256"]} for r in rows]

    def delete_file(self, tenant: str, file_id: str) -> bool:
        row = self.get_file(tenant, file_id)
        if row is None:
            return False
        try:
            Path(row["path"]).unlink(missing_ok=True)
        finally:
            with self._connect() as conn:
                conn.execute("DELETE FROM files WHERE id=? AND tenant=?", (file_id, tenant))
                conn.execute("DELETE FROM vector_store_files WHERE file_id=? AND tenant=?", (file_id, tenant))
        return True

    def create_vector_store(self, tenant: str, req: VectorStoreCreate) -> dict[str, Any]:
        ident = f"vs_{uuid.uuid4().hex}"
        now = self._now()
        expires = now + req.expires_after_seconds if req.expires_after_seconds else None
        with self._connect() as conn:
            conn.execute("INSERT INTO vector_stores VALUES(?,?,?,?,?,?,?)", (ident, tenant, req.name, now, expires, "completed", json.dumps(req.metadata)))
        return self.vector_store(tenant, ident)

    def vector_store(self, tenant: str, ident: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM vector_stores WHERE id=? AND tenant=?", (ident, tenant)).fetchone()
            if row is None:
                return None
            statuses = {r[0]: int(r[1]) for r in conn.execute("SELECT status,count(*) FROM vector_store_files WHERE vector_store_id=? AND tenant=? GROUP BY status", (ident, tenant)).fetchall()}
        total=sum(statuses.values())
        return {"id": row["id"], "object": "vector_store", "name": row["name"], "created_at": row["created_at"],
                "status": "in_progress" if statuses.get("pending",0) or statuses.get("in_progress",0) else ("failed" if statuses.get("failed",0) and not statuses.get("completed",0) else "completed"), "expires_at": row["expires_at"], "metadata": json.loads(row["metadata_json"]),
                "file_counts": {"in_progress": statuses.get("pending",0)+statuses.get("in_progress",0), "completed": statuses.get("completed",0), "failed": statuses.get("failed",0), "cancelled": statuses.get("cancelled",0), "total": total}}

    def list_vector_stores(self, tenant: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            ids = [r[0] for r in conn.execute("SELECT id FROM vector_stores WHERE tenant=? ORDER BY created_at DESC", (tenant,)).fetchall()]
        return [x for ident in ids if (x := self.vector_store(tenant, ident)) is not None]

    def delete_vector_store(self, tenant: str, ident: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM vector_stores WHERE id=? AND tenant=?", (ident, tenant)).fetchone()
            if not row:
                return False
            conn.execute("DELETE FROM vector_store_files WHERE vector_store_id=? AND tenant=?", (ident, tenant))
            conn.execute("DELETE FROM vector_stores WHERE id=? AND tenant=?", (ident, tenant))
        return True

    def attach_vector_file(self, tenant: str, vector_store_id: str, file_id: str) -> dict[str, Any]:
        if self.vector_store(tenant, vector_store_id) is None or self.get_file(tenant, file_id) is None:
            raise KeyError("vector store or file not found")
        now = self._now()
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO vector_store_files VALUES(?,?,?,?,?)", (vector_store_id, file_id, tenant, "pending", now))
        return {"id": file_id, "object": "vector_store.file", "vector_store_id": vector_store_id, "status": "pending", "created_at": now}

    def create_batch(self, tenant: str, req: BatchCreate) -> dict[str, Any]:
        if self.get_file(tenant, req.input_file_id) is None:
            raise KeyError("input file not found")
        ident = f"batch_{uuid.uuid4().hex}"; now = self._now()
        with self._connect() as conn:
            conn.execute("INSERT INTO batches VALUES(?,?,?,?,?,?,?,?,?,?,?)", (ident, tenant, req.input_file_id, req.endpoint, req.completion_window, "validating", now, None, None, None, json.dumps(req.metadata)))
        return self.batch(tenant, ident)

    def batch(self, tenant: str, ident: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM batches WHERE id=? AND tenant=?", (ident, tenant)).fetchone()
        if r is None: return None
        return {"id": r["id"], "object": "batch", "endpoint": r["endpoint"], "input_file_id": r["input_file_id"],
                "completion_window": r["completion_window"], "status": r["status"], "created_at": r["created_at"],
                "cancelled_at": r["cancelled_at"], "output_file_id": r["output_file_id"], "error_file_id": r["error_file_id"],
                "metadata": json.loads(r["metadata_json"]), "request_counts": json.loads(r["metadata_json"]).get("_request_counts", {"total": 0, "completed": 0, "failed": 0})}

    def list_batches(self, tenant: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            ids = [r[0] for r in conn.execute("SELECT id FROM batches WHERE tenant=? ORDER BY created_at DESC", (tenant,)).fetchall()]
        return [x for ident in ids if (x := self.batch(tenant, ident))]

    def cancel_batch(self, tenant: str, ident: str) -> dict[str, Any] | None:
        now = self._now()
        with self._connect() as conn:
            cur = conn.execute("UPDATE batches SET status='cancelled',cancelled_at=? WHERE id=? AND tenant=?", (now, ident, tenant))
        return self.batch(tenant, ident) if cur.rowcount else None

    def create_finetune(self, tenant: str, req: FineTuningJobCreate) -> dict[str, Any]:
        if self.get_file(tenant, req.training_file) is None:
            raise KeyError("training file not found")
        if req.validation_file and self.get_file(tenant, req.validation_file) is None:
            raise KeyError("validation file not found")
        ident = f"ftjob-{uuid.uuid4().hex}"; now = self._now()
        with self._connect() as conn:
            conn.execute("INSERT INTO fine_tuning_jobs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                ident, tenant, req.training_file, req.validation_file, req.model, "queued", now, None, req.suffix,
                json.dumps(req.method) if req.method is not None else None, json.dumps(req.hyperparameters), None, None))
        return self.finetune(tenant, ident)

    def finetune(self, tenant: str, ident: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM fine_tuning_jobs WHERE id=? AND tenant=?", (ident, tenant)).fetchone()
        if r is None: return None
        return {"id": r["id"], "object": "fine_tuning.job", "model": r["model"], "created_at": r["created_at"],
                "finished_at": r["finished_at"], "fine_tuned_model": r["fine_tuned_model"], "status": r["status"],
                "training_file": r["training_file"], "validation_file": r["validation_file"], "suffix": r["suffix"],
                "hyperparameters": json.loads(r["hyperparameters_json"]), "error": json.loads(r["error_json"]) if r["error_json"] else None,
                "method": json.loads(r["method_json"]) if r["method_json"] else None, "trained_tokens": None}

    def list_finetunes(self, tenant: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            ids = [r[0] for r in conn.execute("SELECT id FROM fine_tuning_jobs WHERE tenant=? ORDER BY created_at DESC", (tenant,)).fetchall()]
        return [x for ident in ids if (x := self.finetune(tenant, ident))]

    def cancel_finetune(self, tenant: str, ident: str) -> dict[str, Any] | None:
        now = self._now()
        with self._connect() as conn:
            cur = conn.execute("UPDATE fine_tuning_jobs SET status='cancelled',finished_at=? WHERE id=? AND tenant=? AND status IN ('queued','running','validating_files')", (now, ident, tenant))
        return self.finetune(tenant, ident) if cur.rowcount else self.finetune(tenant, ident)


    # Durable worker control-plane operations. These are intentionally store-level so
    # the same workers work with SQLite and Redis implementations.
    def worker_vector_jobs(self, limit: int = 16) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT tenant,vector_store_id,file_id,status FROM vector_store_files WHERE status IN ('pending','failed') ORDER BY created_at LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def worker_update_vector_file(self, tenant: str, vector_store_id: str, file_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE vector_store_files SET status=? WHERE tenant=? AND vector_store_id=? AND file_id=?", (status, tenant, vector_store_id, file_id))

    def worker_batch_jobs(self, limit: int = 4) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id,tenant FROM batches WHERE status IN ('validating','queued') ORDER BY created_at LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def worker_update_batch(self, tenant: str, ident: str, *, status: str, output_file_id=None, error_file_id=None, request_counts=None, error=None) -> None:
        payload = self.batch(tenant, ident) or {}
        metadata = dict(payload.get('metadata') or {})
        if request_counts is not None: metadata['_request_counts'] = request_counts
        if error is not None: metadata['_worker_error'] = error
        with self._connect() as conn:
            conn.execute("UPDATE batches SET status=?,output_file_id=COALESCE(?,output_file_id),error_file_id=COALESCE(?,error_file_id),metadata_json=? WHERE id=? AND tenant=?", (status, output_file_id, error_file_id, json.dumps(metadata), ident, tenant))

    def worker_finetune_jobs(self, limit: int = 1) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id,tenant FROM fine_tuning_jobs WHERE status='queued' ORDER BY created_at LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def worker_update_finetune(self, tenant: str, ident: str, *, status: str, fine_tuned_model=None, error=None, finished_at=None) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE fine_tuning_jobs SET status=?,fine_tuned_model=COALESCE(?,fine_tuned_model),error_json=?,finished_at=COALESCE(?,finished_at) WHERE id=? AND tenant=?", (status, fine_tuned_model, json.dumps(error) if error is not None else None, finished_at, ident, tenant))

    def get_idempotent(self, tenant: str, route: str, key: str, request_hash: str) -> tuple[int, dict[str, Any]] | None:
        with self._connect() as conn:
            r = conn.execute("SELECT request_hash,status_code,response_json FROM idempotency WHERE tenant=? AND route=? AND idem_key=?", (tenant, route, key)).fetchone()
        if r is None: return None
        if r["request_hash"] != request_hash:
            raise ValueError("idempotency key was already used with a different request")
        return int(r["status_code"]), json.loads(r["response_json"])

    def put_idempotent(self, tenant: str, route: str, key: str, request_hash: str, status_code: int, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO idempotency VALUES(?,?,?,?,?,?,?)", (tenant, route, key, request_hash, status_code, json.dumps(payload), self._now()))


def _tenant(request: Request) -> str:
    principal = getattr(request.state, "principal", None)
    return str(getattr(principal, "tenant_id", None) or request.headers.get("X-Tenant-ID") or "default")


def create_openai_platform_router(store: PlatformStore, *, image_submitter=None, idempotency_store=None, event_dispatch=None) -> APIRouter:
    router = APIRouter()

    def _idem_get(request: Request, route: str, key: str | None, payload: dict):
        if not key or idempotency_store is None:
            return None, None
        digest = idempotency_store.request_hash(payload)
        cached = idempotency_store.get(_tenant(request), route, key, digest)
        return digest, cached

    def _idem_put(request: Request, route: str, key: str | None, digest: str | None, payload: dict):
        if key and digest and idempotency_store is not None:
            idempotency_store.put(_tenant(request), route, key, digest, 200, payload)

    async def _event(event_type: str, data: dict):
        if event_dispatch is not None:
            await event_dispatch({"type": event_type, "created": int(time.time()), "data": data})

    @router.post("/v1/files")
    async def upload_file(request: Request, file: UploadFile = File(...), purpose: str = Form(...)):
        try:
            payload = store.put_file(_tenant(request), file.filename or "upload.bin", purpose, await file.read())
            return payload
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/v1/files")
    async def list_files(request: Request, purpose: str | None = None):
        return {"object": "list", "data": store.list_files(_tenant(request), purpose)}

    @router.get("/v1/files/{file_id}")
    async def retrieve_file(file_id: str, request: Request):
        row = store.get_file(_tenant(request), file_id)
        if row is None: raise HTTPException(404, "file not found")
        return {"id": row["id"], "object": "file", "bytes": row["bytes"], "created_at": row["created_at"], "filename": row["filename"], "purpose": row["purpose"], "status": "processed", "sha256": row["sha256"]}

    @router.get("/v1/files/{file_id}/content")
    async def file_content(file_id: str, request: Request):
        row = store.get_file(_tenant(request), file_id)
        if row is None: raise HTTPException(404, "file not found")
        return FileResponse(row["path"], filename=row["filename"])

    @router.delete("/v1/files/{file_id}")
    async def delete_file(file_id: str, request: Request):
        deleted = store.delete_file(_tenant(request), file_id)
        return {"id": file_id, "object": "file", "deleted": deleted}

    @router.post("/v1/vector_stores")
    async def create_vs(body: VectorStoreCreate, request: Request, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        digest, cached = _idem_get(request, "/v1/vector_stores", idempotency_key, body.model_dump(mode="json"))
        if cached is not None: return cached[1]
        result = store.create_vector_store(_tenant(request), body); _idem_put(request, "/v1/vector_stores", idempotency_key, digest, result)
        await _event("vector_store.created", result); return result
    @router.get("/v1/vector_stores")
    async def list_vs(request: Request): return {"object": "list", "data": store.list_vector_stores(_tenant(request))}
    @router.get("/v1/vector_stores/{vector_store_id}")
    async def get_vs(vector_store_id: str, request: Request):
        value = store.vector_store(_tenant(request), vector_store_id)
        if value is None: raise HTTPException(404, "vector store not found")
        return value
    @router.delete("/v1/vector_stores/{vector_store_id}")
    async def delete_vs(vector_store_id: str, request: Request): return {"id": vector_store_id, "object": "vector_store.deleted", "deleted": store.delete_vector_store(_tenant(request), vector_store_id)}
    @router.post("/v1/vector_stores/{vector_store_id}/files")
    async def attach_vs_file(vector_store_id: str, body: VectorStoreFileCreate, request: Request):
        try: return store.attach_vector_file(_tenant(request), vector_store_id, body.file_id)
        except KeyError as exc: raise HTTPException(404, str(exc)) from exc

    @router.post("/v1/batches")
    async def create_batch(body: BatchCreate, request: Request, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        digest, cached = _idem_get(request, "/v1/batches", idempotency_key, body.model_dump(mode="json"))
        if cached is not None: return cached[1]
        try: result = store.create_batch(_tenant(request), body)
        except KeyError as exc: raise HTTPException(404, str(exc)) from exc
        _idem_put(request, "/v1/batches", idempotency_key, digest, result); await _event("batch.created", result); return result
    @router.get("/v1/batches")
    async def list_batches(request: Request): return {"object": "list", "data": store.list_batches(_tenant(request))}
    @router.get("/v1/batches/{batch_id}")
    async def get_batch(batch_id: str, request: Request):
        value = store.batch(_tenant(request), batch_id)
        if value is None: raise HTTPException(404, "batch not found")
        return value
    @router.post("/v1/batches/{batch_id}/cancel")
    async def cancel_batch(batch_id: str, request: Request):
        value = store.cancel_batch(_tenant(request), batch_id)
        if value is None: raise HTTPException(404, "batch not found")
        await _event("batch.cancelled", value); return value

    @router.post("/v1/fine_tuning/jobs")
    async def create_ft(body: FineTuningJobCreate, request: Request, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        digest, cached = _idem_get(request, "/v1/fine_tuning/jobs", idempotency_key, body.model_dump(mode="json"))
        if cached is not None: return cached[1]
        try: result = store.create_finetune(_tenant(request), body)
        except KeyError as exc: raise HTTPException(404, str(exc)) from exc
        _idem_put(request, "/v1/fine_tuning/jobs", idempotency_key, digest, result); await _event("fine_tuning.job.created", result); return result
    @router.get("/v1/fine_tuning/jobs")
    async def list_ft(request: Request): return {"object": "list", "data": store.list_finetunes(_tenant(request)), "has_more": False}
    @router.get("/v1/fine_tuning/jobs/{job_id}")
    async def get_ft(job_id: str, request: Request):
        value = store.finetune(_tenant(request), job_id)
        if value is None: raise HTTPException(404, "fine-tuning job not found")
        return value
    @router.post("/v1/fine_tuning/jobs/{job_id}/cancel")
    async def cancel_ft(job_id: str, request: Request):
        value = store.cancel_finetune(_tenant(request), job_id)
        if value is None: raise HTTPException(404, "fine-tuning job not found")
        await _event("fine_tuning.job.cancelled", value); return value

    @router.post("/v1/images/generations")
    async def images(body: ImageGenerationRequest, request: Request, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        if image_submitter is None:
            raise HTTPException(503, "image generation backend is not configured")
        req_payload = body.model_dump(mode="json")
        try:
            digest, cached = _idem_get(request, "/v1/images/generations", idempotency_key, req_payload)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if cached is not None: return cached[1]
        result = await image_submitter(body, request)
        _idem_put(request, "/v1/images/generations", idempotency_key, digest, result)
        await _event("image.generation.completed", result); return result

    return router


__all__ = ["PlatformStore", "create_openai_platform_router", "ImageGenerationRequest"]
