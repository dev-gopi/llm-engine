"""Opt-in semantic and episodic memory with explicit retention/deletion controls."""
from __future__ import annotations
from dataclasses import dataclass
import sqlite3
import time
from pathlib import Path
from typing import Sequence

from evaluation.embeddings import HashEmbeddingModel

@dataclass(frozen=True)
class MemoryRecord:
    memory_id: int
    user_id: str
    kind: str
    content: str
    created_at: float
    expires_at: float | None

class LongTermMemory:
    def __init__(self, path: str | Path, *, embedding_model=None, retention_seconds: int | None = None):
        if retention_seconds is not None and retention_seconds <= 0: raise ValueError("retention_seconds must be positive")
        self.path=str(path); self.embedding_model=embedding_model or HashEmbeddingModel(); self.retention_seconds=retention_seconds
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, kind TEXT NOT NULL, content TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL, embedding BLOB NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)")

    def add(self, user_id: str, content: str, *, kind: str = "episodic") -> int:
        if not user_id or not content.strip(): raise ValueError("user_id and content are required")
        if kind not in {"episodic","semantic"}: raise ValueError("kind must be episodic or semantic")
        now=time.time(); expiry=now+self.retention_seconds if self.retention_seconds else None
        vector=self.embedding_model.encode([content])[0].tolist()
        import json
        with sqlite3.connect(self.path) as db:
            cur=db.execute("INSERT INTO memories(user_id,kind,content,created_at,expires_at,embedding) VALUES(?,?,?,?,?,?)",(user_id,kind,content,now,expiry,json.dumps(vector)))
            return int(cur.lastrowid)

    def retrieve(self, user_id: str, query: str, *, limit: int = 5) -> list[MemoryRecord]:
        if not user_id or not query.strip() or limit < 1: raise ValueError("invalid memory query")
        import json, torch
        q=self.embedding_model.encode([query])[0]
        now=time.time(); rows=[]
        with sqlite3.connect(self.path) as db:
            for row in db.execute("SELECT id,user_id,kind,content,created_at,expires_at,embedding FROM memories WHERE user_id=? AND (expires_at IS NULL OR expires_at>?)",(user_id,now)):
                vector=torch.tensor(json.loads(row[6]),dtype=torch.float32)
                score=float(torch.nn.functional.cosine_similarity(q,vector,dim=0))
                rows.append((score,MemoryRecord(row[0],row[1],row[2],row[3],row[4],row[5])))
        rows.sort(key=lambda x:(x[0],x[1].memory_id),reverse=True)
        return [record for score,record in rows[:limit] if score > 0.25]

    def delete_user(self, user_id: str) -> int:
        with sqlite3.connect(self.path) as db:
            return int(db.execute("DELETE FROM memories WHERE user_id=?",(user_id,)).rowcount)

    def purge_expired(self) -> int:
        with sqlite3.connect(self.path) as db:
            return int(db.execute("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at<=?",(time.time(),)).rowcount)
