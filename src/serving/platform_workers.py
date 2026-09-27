"""Durable execution workers for Vector Stores, Batch and Fine-tuning Jobs."""

from __future__ import annotations

import asyncio
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from embeddings import EmbeddingService


def _text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {
        ".txt",
        ".md",
        ".py",
        ".json",
        ".jsonl",
        ".csv",
        ".yaml",
        ".yml",
        ".html",
        ".htm",
    }:
        return path.read_text("utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            return "\n".join(
                (p.extract_text() or "") for p in PdfReader(str(path)).pages
            )
        except ImportError as exc:
            raise RuntimeError("install llm-engine[rag] for PDF ingestion") from exc
    if suffix == ".docx":
        try:
            from docx import Document

            return "\n".join(p.text for p in Document(str(path)).paragraphs)
        except ImportError as exc:
            raise RuntimeError("install llm-engine[rag] for DOCX ingestion") from exc
    raise ValueError(f"unsupported vector-store file type: {suffix or '<none>'}")


def _chunks(text: str, size: int = 1000, overlap: int = 150):
    text = " ".join(text.split())
    if not text:
        return []
    out = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        out.append(text[start:end])
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return out


class VectorIndex:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as c:
            c.executescript(
                """CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY,tenant TEXT,vector_store_id TEXT,file_id TEXT,ordinal INTEGER,text TEXT,embedding_json TEXT,created_at INTEGER); CREATE INDEX IF NOT EXISTS chunks_vs ON chunks(tenant,vector_store_id);"""
            )
        self.emb = EmbeddingService()

    def replace_file(self, tenant, vs, file_id, texts):
        vectors = self.emb.encode(texts, normalize=True).embeddings if texts else []
        now = int(time.time())
        with sqlite3.connect(self.path) as c:
            c.execute(
                "DELETE FROM chunks WHERE tenant=? AND vector_store_id=? AND file_id=?",
                (tenant, vs, file_id),
            )
            for i, (t, v) in enumerate(zip(texts, vectors)):
                c.execute(
                    "INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)",
                    (
                        f"ch_{uuid.uuid4().hex}",
                        tenant,
                        vs,
                        file_id,
                        i,
                        t,
                        json.dumps(list(map(float, v))),
                        now,
                    ),
                )
        return len(texts)

    def search(self, tenant, vs, query, top_k=5):
        q = list(map(float, self.emb.encode([query], normalize=True).embeddings[0]))
        qn = math.sqrt(sum(x * x for x in q)) or 1
        with sqlite3.connect(self.path) as c:
            rows = c.execute(
                "SELECT id,file_id,ordinal,text,embedding_json FROM chunks WHERE tenant=? AND vector_store_id=?",
                (tenant, vs),
            ).fetchall()
        scored = []
        for ident, fid, ord_, text, raw in rows:
            v = json.loads(raw)
            vn = math.sqrt(sum(x * x for x in v)) or 1
            score = sum(a * b for a, b in zip(q, v)) / (qn * vn)
            scored.append(
                {
                    "id": ident,
                    "file_id": fid,
                    "ordinal": ord_,
                    "text": text,
                    "score": score,
                }
            )
        return sorted(scored, key=lambda x: x["score"], reverse=True)[: max(1, top_k)]


@dataclass
class WorkerResult:
    processed: int = 0
    succeeded: int = 0
    failed: int = 0


class PlatformWorkers:
    def __init__(
        self,
        store,
        *,
        vector_index_path="data/cache/vector-index.sqlite3",
        batch_executor: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
        | None = None,
        finetune_command_builder=None,
    ):
        self.store = store
        self.vector_index = VectorIndex(vector_index_path)
        self.batch_executor = batch_executor
        self.finetune_command_builder = (
            finetune_command_builder or self._default_ft_command
        )

    def ingest_vector_file(self, tenant, vector_store_id, file_id):
        row = self.store.get_file(tenant, file_id)
        if row is None:
            raise KeyError("file not found")
        path = Path(row["path"])
        self.store.worker_update_vector_file(
            tenant, vector_store_id, file_id, "in_progress"
        )
        try:
            count = self.vector_index.replace_file(
                tenant, vector_store_id, file_id, _chunks(_text_from_file(path))
            )
            self.store.worker_update_vector_file(
                tenant, vector_store_id, file_id, "completed"
            )
            return count
        except Exception:
            self.store.worker_update_vector_file(
                tenant, vector_store_id, file_id, "failed"
            )
            raise

    async def run_vector_once(self, limit=16):
        r = WorkerResult()
        for j in self.store.worker_vector_jobs(limit):
            r.processed += 1
            try:
                self.ingest_vector_file(j["tenant"], j["vector_store_id"], j["file_id"])
                r.succeeded += 1
            except Exception:
                r.failed += 1
        return r

    async def execute_batch(self, tenant, batch_id):
        job = self.store.batch(tenant, batch_id)
        if not job:
            raise KeyError("batch not found")
        inp = self.store.get_file(tenant, job["input_file_id"])
        lines = Path(inp["path"]).read_text("utf-8").splitlines()
        self.store.worker_update_batch(
            tenant,
            batch_id,
            status="in_progress",
            request_counts={"total": len(lines), "completed": 0, "failed": 0},
        )
        outs = []
        errs = []
        completed = failed = 0
        for n, line in enumerate(lines, 1):
            try:
                item = json.loads(line)
                custom_id = item.get("custom_id", str(n))
                endpoint = item.get("url") or job["endpoint"]
                body = item.get("body", {})
                if endpoint != job["endpoint"]:
                    raise ValueError("request endpoint does not match batch endpoint")
                if self.batch_executor is None:
                    response = {"status_code": 200, "body": body}
                else:
                    response = await self.batch_executor(endpoint, body)
                outs.append(
                    json.dumps(
                        {
                            "id": f"batch_req_{uuid.uuid4().hex}",
                            "custom_id": custom_id,
                            "response": response,
                            "error": None,
                        }
                    )
                )
                completed += 1
            except Exception as exc:
                errs.append(
                    json.dumps(
                        {
                            "line": n,
                            "error": {"message": str(exc), "type": type(exc).__name__},
                        }
                    )
                )
                failed += 1
        out_id = (
            self.store.put_file(
                tenant,
                f"{batch_id}-output.jsonl",
                "batch_output",
                ("\n".join(outs) + "\n").encode(),
            )["id"]
            if outs
            else None
        )
        err_id = (
            self.store.put_file(
                tenant,
                f"{batch_id}-errors.jsonl",
                "batch_error",
                ("\n".join(errs) + "\n").encode(),
            )["id"]
            if errs
            else None
        )
        self.store.worker_update_batch(
            tenant,
            batch_id,
            status="completed" if completed else "failed",
            output_file_id=out_id,
            error_file_id=err_id,
            request_counts={
                "total": len(lines),
                "completed": completed,
                "failed": failed,
            },
        )
        return self.store.batch(tenant, batch_id)

    async def run_batch_once(self, limit=4):
        r = WorkerResult()
        for j in self.store.worker_batch_jobs(limit):
            r.processed += 1
            try:
                await self.execute_batch(j["tenant"], j["id"])
                r.succeeded += 1
            except Exception as exc:
                self.store.worker_update_batch(
                    j["tenant"], j["id"], status="failed", error={"message": str(exc)}
                )
                r.failed += 1
        return r

    def _default_ft_command(self, job, train_path, val_path, out_dir):
        import yaml

        hp = job.get("hyperparameters") or {}
        base = Path(str(hp.get("training_config", "configs/finetuning.gpu.yaml")))
        data = yaml.safe_load(base.read_text("utf-8")) or {}
        data["train_files"] = [str(train_path)]
        data["validation_files"] = [str(val_path)] if val_path else []
        data["samples_per_epoch"] = (
            int(hp.get("samples_per_epoch", data.get("samples_per_epoch", 0))) or None
        )
        if data.get("samples_per_epoch") is None:
            data.pop("samples_per_epoch", None)
        for key in (
            "batch_size",
            "validation_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "epochs",
            "max_sequence_length",
        ):
            if key in hp:
                data[key] = hp[key]
        generated = out_dir / "worker-training.yaml"
        generated.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        cmd = [
            sys.executable,
            "scripts/train.py",
            "--training-config",
            str(generated),
            "--output",
            str(out_dir / "model.pt"),
            "--best-output",
            str(out_dir / "best.pt"),
            "--report-json",
            str(out_dir / "report.json"),
            "--log-file",
            str(out_dir / "training.log"),
            "--no-live-report",
        ]
        if hp.get("model_config"):
            cmd += ["--model-config", str(hp["model_config"])]
        if hp.get("tokenizer"):
            cmd += ["--tokenizer", str(hp["tokenizer"])]
        if hp.get("init_from"):
            cmd += ["--init-from", str(hp["init_from"])]
        if hp.get("epochs"):
            cmd += ["--epochs", str(hp["epochs"])]
        return cmd

    def execute_finetune(self, tenant, job_id):
        job = self.store.finetune(tenant, job_id)
        if not job:
            raise KeyError("fine-tuning job not found")
        train = Path(self.store.get_file(tenant, job["training_file"])["path"])
        val = (
            Path(self.store.get_file(tenant, job["validation_file"])["path"])
            if job.get("validation_file")
            else None
        )
        out = (
            Path(os.getenv("GOPI_FINETUNE_OUTPUT_ROOT", "outputs/fine_tuning")) / job_id
        )
        out.mkdir(parents=True, exist_ok=True)
        cmd = self.finetune_command_builder(job, train, val, out)
        self.store.worker_update_finetune(tenant, job_id, status="running")
        try:
            p = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=float(os.getenv("GOPI_FINETUNE_TIMEOUT_SECONDS", "86400")),
                check=False,
            )
            (out / "worker.stdout.log").write_text(p.stdout or "", encoding="utf-8")
            (out / "worker.stderr.log").write_text(p.stderr or "", encoding="utf-8")
            if p.returncode:
                raise RuntimeError(f"training process exited {p.returncode}")
            model = f"ft:{job['model']}:{job_id}"
            self.store.worker_update_finetune(
                tenant,
                job_id,
                status="succeeded",
                fine_tuned_model=model,
                finished_at=int(time.time()),
            )
            return self.store.finetune(tenant, job_id)
        except Exception as exc:
            self.store.worker_update_finetune(
                tenant,
                job_id,
                status="failed",
                error={"message": str(exc), "type": type(exc).__name__},
                finished_at=int(time.time()),
            )
            raise

    async def run_finetune_once(self, limit=1):
        r = WorkerResult()
        for j in self.store.worker_finetune_jobs(limit):
            r.processed += 1
            try:
                await asyncio.to_thread(self.execute_finetune, j["tenant"], j["id"])
                r.succeeded += 1
            except Exception:
                r.failed += 1
        return r


__all__ = ["VectorIndex", "PlatformWorkers", "WorkerResult"]
