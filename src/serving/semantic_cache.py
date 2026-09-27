"""Safe semantic response caching for deterministic serving requests.

The cache is deliberately conservative: requests with sessions, tools, RAG,
web search, attachments, reasoning, or non-deterministic sampling are bypassed
by default. Exact-request identity and semantic similarity are separated so a
semantic hit is only possible when every non-prompt generation parameter is
identical.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from embeddings import EmbeddingService

from .runtime import BackendGeneration
from .schemas import FinishReason, GenerateRequest


@dataclass(frozen=True)
class SemanticCacheConfig:
    enabled: bool = False
    capacity: int = 512
    similarity_threshold: float = 0.985
    ttl_seconds: float = 3600.0
    namespace: str = "default"
    sqlite_path: str | None = None
    allow_nondeterministic: bool = False

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("semantic cache capacity must be positive")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                "semantic cache similarity_threshold must be between 0 and 1"
            )
        if self.ttl_seconds <= 0:
            raise ValueError("semantic cache ttl_seconds must be positive")
        if not self.namespace.strip():
            raise ValueError("semantic cache namespace cannot be empty")


@dataclass
class _CacheEntry:
    exact_key: str
    policy_key: str
    prompt: str
    vector: tuple[float, ...]
    result: BackendGeneration
    created_at: float
    expires_at: float
    last_access: float


class SemanticResponseCache:
    """Two-tier exact + semantic cache with optional SQLite persistence."""

    def __init__(
        self,
        config: SemanticCacheConfig,
        *,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.config = config
        self.embedding_service = embedding_service or EmbeddingService()
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self._db_lock = threading.RLock()
        self._db: sqlite3.Connection | None = None
        self._metrics = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "bypasses": 0,
            "stores": 0,
            "evictions": 0,
            "expired": 0,
        }
        if config.sqlite_path:
            path = Path(config.sqlite_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_response_cache (
                    namespace TEXT NOT NULL,
                    exact_key TEXT NOT NULL,
                    policy_key TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    last_access REAL NOT NULL,
                    PRIMARY KEY(namespace, exact_key)
                )
                """
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS semantic_response_cache_policy_idx "
                "ON semantic_response_cache(namespace, policy_key, expires_at)"
            )
            self._db.commit()

    def close(self) -> None:
        if self._db is not None:
            with self._db_lock:
                self._db.close()
                self._db = None

    def eligible(self, request: GenerateRequest) -> bool:
        if not self.config.enabled:
            return False
        dynamic = bool(
            request.session_id
            or request.tools
            or request.chat_tools
            or request.mcp
            or request.web_search
            or request.rag
            or request.attachments
            or request.reasoning_effort != "none"
        )
        if dynamic:
            return False
        if not self.config.allow_nondeterministic and request.temperature != 0.0:
            return False
        return True

    def request_key(self, request: GenerateRequest) -> str:
        return self._hash_payload(self._request_payload(request))

    def policy_key(self, request: GenerateRequest) -> str:
        payload = self._request_payload(request)
        request_payload = dict(payload["request"])
        request_payload.pop("prompt", None)
        payload = {"namespace": payload["namespace"], "request": request_payload}
        return self._hash_payload(payload)

    def lock_for(self, request: GenerateRequest) -> asyncio.Lock:
        key = self.request_key(request)
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    def lookup(self, request: GenerateRequest) -> BackendGeneration | None:
        if not self.eligible(request):
            self._metrics["bypasses"] += 1
            return None
        now = time.time()
        self._purge_expired(now)
        exact_key = self.request_key(request)
        policy_key = self.policy_key(request)

        entry = self._entries.get(exact_key)
        if entry is None and self._db is not None:
            entry = self._load_exact(exact_key, now)
        if entry is not None:
            self._touch(entry, now)
            self._metrics["exact_hits"] += 1
            return self._cached_result(entry.result)

        query_vector = self._embed(request.prompt)
        best_entry: _CacheEntry | None = None
        best_score = -1.0
        candidates = [
            entry for entry in self._entries.values() if entry.policy_key == policy_key
        ]
        if self._db is not None:
            known = {entry.exact_key for entry in candidates}
            candidates.extend(
                entry
                for entry in self._load_policy(policy_key, now)
                if entry.exact_key not in known
            )
        for candidate in candidates:
            score = self._cosine(query_vector, candidate.vector)
            if score > best_score:
                best_entry, best_score = candidate, score
        if best_entry is not None and best_score >= self.config.similarity_threshold:
            self._touch(best_entry, now)
            self._metrics["semantic_hits"] += 1
            return self._cached_result(best_entry.result)

        self._metrics["misses"] += 1
        return None

    def store(self, request: GenerateRequest, result: BackendGeneration) -> bool:
        if not self.eligible(request):
            return False
        if result.finish_reason not in {FinishReason.STOP, FinishReason.LENGTH}:
            return False
        if result.tool_calls or result.tool_call_error or result.reasoning_content:
            return False
        now = time.time()
        entry = _CacheEntry(
            exact_key=self.request_key(request),
            policy_key=self.policy_key(request),
            prompt=request.prompt,
            vector=self._embed(request.prompt),
            result=result,
            created_at=now,
            expires_at=now + self.config.ttl_seconds,
            last_access=now,
        )
        self._entries[entry.exact_key] = entry
        self._entries.move_to_end(entry.exact_key)
        self._write_entry(entry)
        self._metrics["stores"] += 1
        self._enforce_capacity()
        return True

    def purge(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        if self._db is not None:
            with self._db_lock:
                row = self._db.execute(
                    "SELECT COUNT(*) FROM semantic_response_cache WHERE namespace = ?",
                    (self.config.namespace,),
                ).fetchone()
                count = max(count, int(row[0] if row else 0))
                self._db.execute(
                    "DELETE FROM semantic_response_cache WHERE namespace = ?",
                    (self.config.namespace,),
                )
                self._db.commit()
        return count

    def metrics(self) -> dict[str, int | float | str]:
        lookups = (
            self._metrics["exact_hits"]
            + self._metrics["semantic_hits"]
            + self._metrics["misses"]
        )
        hits = self._metrics["exact_hits"] + self._metrics["semantic_hits"]
        return {
            **self._metrics,
            "entries": len(self._entries),
            "hit_rate": hits / lookups if lookups else 0.0,
            "namespace": self.config.namespace,
            "similarity_threshold": self.config.similarity_threshold,
        }

    def _request_payload(self, request: GenerateRequest) -> dict[str, Any]:
        payload = request.model_dump(mode="json")
        # Semantic cache identity must never depend on private runtime state.
        return {"namespace": self.config.namespace, "request": payload}

    @staticmethod
    def _hash_payload(payload: dict[str, Any]) -> str:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _embed(self, text: str) -> tuple[float, ...]:
        vector = self.embedding_service.encode([text], normalize=True).embeddings[0]
        return tuple(float(value) for value in vector)

    @staticmethod
    def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        if len(left) != len(right) or not left:
            return -1.0
        value = sum(a * b for a, b in zip(left, right))
        return max(-1.0, min(1.0, value)) if math.isfinite(value) else -1.0

    @staticmethod
    def _cached_result(result: BackendGeneration) -> BackendGeneration:
        return BackendGeneration(
            text=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finish_reason=result.finish_reason,
            cached_tokens=max(result.cached_tokens, result.prompt_tokens),
            reasoning_tokens=result.reasoning_tokens,
            structured_output_valid=result.structured_output_valid,
            structured_output_error=result.structured_output_error,
            reasoning_content=result.reasoning_content,
            tool_calls=result.tool_calls,
            tool_call_error=result.tool_call_error,
            logprobs=result.logprobs,
        )

    def _touch(self, entry: _CacheEntry, now: float) -> None:
        entry.last_access = now
        self._entries[entry.exact_key] = entry
        self._entries.move_to_end(entry.exact_key)
        if self._db is not None:
            with self._db_lock:
                self._db.execute(
                    "UPDATE semantic_response_cache SET last_access = ? WHERE namespace = ? AND exact_key = ?",
                    (now, self.config.namespace, entry.exact_key),
                )
                self._db.commit()

    def _purge_expired(self, now: float) -> None:
        expired = [
            key for key, entry in self._entries.items() if entry.expires_at <= now
        ]
        for key in expired:
            self._entries.pop(key, None)
        if expired:
            self._metrics["expired"] += len(expired)
        if self._db is not None:
            with self._db_lock:
                row = self._db.execute(
                    "SELECT COUNT(*) FROM semantic_response_cache WHERE namespace = ? AND expires_at <= ?",
                    (self.config.namespace, now),
                ).fetchone()
                db_expired = int(row[0] if row else 0)
                self._db.execute(
                    "DELETE FROM semantic_response_cache WHERE namespace = ? AND expires_at <= ?",
                    (self.config.namespace, now),
                )
                self._db.commit()
                self._metrics["expired"] += max(0, db_expired - len(expired))

    def _enforce_capacity(self) -> None:
        while len(self._entries) > self.config.capacity:
            key, _ = self._entries.popitem(last=False)
            self._metrics["evictions"] += 1
            if self._db is not None:
                with self._db_lock:
                    self._db.execute(
                        "DELETE FROM semantic_response_cache WHERE namespace = ? AND exact_key = ?",
                        (self.config.namespace, key),
                    )
                    self._db.commit()

    def _write_entry(self, entry: _CacheEntry) -> None:
        if self._db is None:
            return
        payload = {
            **asdict(entry.result),
            "finish_reason": entry.result.finish_reason.value,
            "tool_calls": [
                call.model_dump(mode="json") for call in entry.result.tool_calls
            ],
        }
        with self._db_lock:
            self._db.execute(
                """
                INSERT OR REPLACE INTO semantic_response_cache
                (namespace, exact_key, policy_key, prompt, vector_json, result_json, created_at, expires_at, last_access)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.config.namespace,
                    entry.exact_key,
                    entry.policy_key,
                    entry.prompt,
                    json.dumps(entry.vector),
                    json.dumps(payload),
                    entry.created_at,
                    entry.expires_at,
                    entry.last_access,
                ),
            )
            self._db.commit()

    def _load_exact(self, exact_key: str, now: float) -> _CacheEntry | None:
        if self._db is None:
            return None
        with self._db_lock:
            row = self._db.execute(
                """SELECT exact_key, policy_key, prompt, vector_json, result_json, created_at, expires_at, last_access
                   FROM semantic_response_cache
                   WHERE namespace = ? AND exact_key = ? AND expires_at > ?""",
                (self.config.namespace, exact_key, now),
            ).fetchone()
        return self._row_to_entry(row) if row else None

    def _load_policy(self, policy_key: str, now: float) -> list[_CacheEntry]:
        if self._db is None:
            return []
        with self._db_lock:
            rows = self._db.execute(
                """SELECT exact_key, policy_key, prompt, vector_json, result_json, created_at, expires_at, last_access
                   FROM semantic_response_cache
                   WHERE namespace = ? AND policy_key = ? AND expires_at > ?
                   ORDER BY last_access DESC LIMIT ?""",
                (self.config.namespace, policy_key, now, self.config.capacity),
            ).fetchall()
        entries = [self._row_to_entry(row) for row in rows]
        for entry in entries:
            self._entries[entry.exact_key] = entry
            self._entries.move_to_end(entry.exact_key)
        self._enforce_capacity()
        return entries

    @staticmethod
    def _row_to_entry(row: tuple[Any, ...]) -> _CacheEntry:
        (
            exact_key,
            policy_key,
            prompt,
            vector_json,
            result_json,
            created_at,
            expires_at,
            last_access,
        ) = row
        payload = json.loads(result_json)
        payload["finish_reason"] = FinishReason(
            payload.get("finish_reason", FinishReason.STOP.value)
        )
        payload["tool_calls"] = ()
        return _CacheEntry(
            exact_key=exact_key,
            policy_key=policy_key,
            prompt=prompt,
            vector=tuple(float(value) for value in json.loads(vector_json)),
            result=BackendGeneration(**payload),
            created_at=float(created_at),
            expires_at=float(expires_at),
            last_access=float(last_access),
        )
