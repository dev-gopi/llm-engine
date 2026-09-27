"""Reusable idempotency-key storage for mutation endpoints."""
from __future__ import annotations
import hashlib, json, sqlite3, time
from pathlib import Path
from typing import Any

class IdempotencyConflict(ValueError): pass

class IdempotencyStore:
    def __init__(self, path: str | Path, ttl_seconds: int = 86400) -> None:
        self.path=Path(path); self.path.parent.mkdir(parents=True, exist_ok=True); self.ttl_seconds=max(60,int(ttl_seconds))
        with sqlite3.connect(self.path) as c:
            c.execute("CREATE TABLE IF NOT EXISTS idempotency(tenant TEXT, route TEXT, key TEXT, request_hash TEXT, status INTEGER, body TEXT, created_at INTEGER, PRIMARY KEY(tenant,route,key))")
    @staticmethod
    def request_hash(payload: Any) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",",":"), default=str).encode()).hexdigest()
    def get(self, tenant:str, route:str, key:str, request_hash:str):
        now=int(time.time())
        with sqlite3.connect(self.path) as c:
            c.row_factory=sqlite3.Row
            c.execute("DELETE FROM idempotency WHERE created_at<?", (now-self.ttl_seconds,))
            r=c.execute("SELECT * FROM idempotency WHERE tenant=? AND route=? AND key=?",(tenant,route,key)).fetchone()
        if r is None: return None
        if r["request_hash"] != request_hash: raise IdempotencyConflict("idempotency key reused with a different request")
        return int(r["status"]), json.loads(r["body"])
    def put(self, tenant:str, route:str, key:str, request_hash:str, status:int, body:dict)->None:
        with sqlite3.connect(self.path) as c:
            c.execute("INSERT OR REPLACE INTO idempotency VALUES(?,?,?,?,?,?,?)",(tenant,route,key,request_hash,int(status),json.dumps(body),int(time.time())))
