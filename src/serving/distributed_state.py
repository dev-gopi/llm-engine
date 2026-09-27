"""Shared Redis-backed serving state for horizontally scaled deployments.

The adapters intentionally mirror the existing SQLite/process-local contracts so
operators can switch stores with configuration rather than changing request code.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

from inference.context import ConversationMemory
from serving.idempotency import IdempotencyConflict
from serving.rate_limit import RateLimitDecision

try:  # optional dependency
    import redis
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover
    redis = None
    aioredis = None


class RedisRateLimiter:
    """Atomic cross-host fixed-window limiter implemented with a Redis ZSET."""

    def __init__(
        self, url: str, requests_per_minute: int, *, key_prefix: str = "llm-engine:rate"
    ) -> None:
        if aioredis is None:
            raise RuntimeError("redis package is required for RedisRateLimiter")
        self.limit = int(requests_per_minute)
        self.key_prefix = key_prefix.rstrip(":")
        self.client = aioredis.from_url(url, decode_responses=True)

    def _key(self, identity: str) -> str:
        digest = hashlib.sha256(identity.encode()).hexdigest()
        return f"{self.key_prefix}:{digest}"

    async def allow(self, identity: str) -> bool:
        return (await self.check(identity)).allowed

    async def check(self, identity: str) -> RateLimitDecision:
        if self.limit <= 0:
            return RateLimitDecision(True, self.limit, self.limit, 0.0)
        now = time.time()
        cutoff = now - 60.0
        member = f"{now:.9f}:{id(asyncio.current_task())}:{time.monotonic_ns()}"
        script = """
        redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
        local n=redis.call('ZCARD', KEYS[1])
        if n >= tonumber(ARGV[3]) then
          local first=redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
          local oldest=tonumber(first[2] or ARGV[2])
          return {0,n,oldest}
        end
        redis.call('ZADD', KEYS[1], ARGV[2], ARGV[4])
        redis.call('EXPIRE', KEYS[1], 61)
        return {1,n+1,0}
        """
        allowed, count, oldest = await self.client.eval(
            script, 1, self._key(identity), cutoff, now, self.limit, member
        )
        allowed = bool(int(allowed))
        count = int(count)
        retry = max(0.0, 60.0 - (now - float(oldest))) if not allowed else 0.0
        return RateLimitDecision(allowed, self.limit, max(self.limit - count, 0), retry)

    async def close(self) -> None:
        close = getattr(self.client, "aclose", None)
        if close is not None:
            await close()


class RedisIdempotencyStore:
    """Cross-host idempotency store with request-hash conflict protection."""

    def __init__(
        self,
        url: str,
        *,
        ttl_seconds: int = 86400,
        key_prefix: str = "llm-engine:idempotency",
    ) -> None:
        if redis is None:
            raise RuntimeError("redis package is required for RedisIdempotencyStore")
        self.client = redis.from_url(url, decode_responses=True)
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.key_prefix = key_prefix.rstrip(":")

    @staticmethod
    def request_hash(payload: Any) -> str:
        return hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), default=str
            ).encode()
        ).hexdigest()

    def _key(self, tenant: str, route: str, key: str) -> str:
        digest = hashlib.sha256(f"{tenant}\0{route}\0{key}".encode()).hexdigest()
        return f"{self.key_prefix}:{digest}"

    def get(self, tenant: str, route: str, key: str, request_hash: str):
        raw = self.client.get(self._key(tenant, route, key))
        if raw is None:
            return None
        value = json.loads(raw)
        if value.get("request_hash") != request_hash:
            raise IdempotencyConflict("idempotency key reused with a different request")
        return int(value["status"]), value["body"]

    def put(
        self,
        tenant: str,
        route: str,
        key: str,
        request_hash: str,
        status: int,
        body: dict,
    ) -> None:
        redis_key = self._key(tenant, route, key)
        payload = json.dumps(
            {"request_hash": request_hash, "status": int(status), "body": body},
            separators=(",", ":"),
        )
        script = """
        local old=redis.call('GET', KEYS[1])
        if old then
          local obj=cjson.decode(old)
          if obj.request_hash ~= ARGV[1] then return -1 end
        end
        redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3]); return 1
        """
        result = int(
            self.client.eval(
                script, 1, redis_key, request_hash, payload, self.ttl_seconds
            )
        )
        if result < 0:
            raise IdempotencyConflict("idempotency key reused with a different request")


class RedisSessionStore:
    """Shared Redis implementation of the SQLiteSessionStore public lifecycle."""

    def __init__(
        self,
        url: str,
        tokenizer,
        *,
        max_tokens: int,
        system_prompt: str,
        ttl_seconds: int = 86400,
        key_prefix: str = "llm-engine:sessions",
    ) -> None:
        if redis is None:
            raise RuntimeError("redis package is required for RedisSessionStore")
        self.client = redis.from_url(url, decode_responses=True)
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.key_prefix = key_prefix.rstrip(":")

    def _session(self, sid: str) -> str:
        return f"{self.key_prefix}:session:{sid}"

    def _index(self) -> str:
        return f"{self.key_prefix}:index"

    def _approved(self, sid: str) -> str:
        return f"{self.key_prefix}:approved:{sid}"

    def load(self, session_id: str) -> ConversationMemory:
        memory = ConversationMemory(
            self.tokenizer, max_tokens=self.max_tokens, system_prompt=self.system_prompt
        )
        raw = self.client.get(self._session(session_id))
        if raw:
            memory.restore(json.loads(raw))
            self.client.expire(self._session(session_id), self.ttl_seconds)
        return memory

    def save(self, session_id: str, memory: ConversationMemory) -> None:
        payload = json.dumps(
            [{"role": m.role, "content": m.content} for m in memory.snapshot()],
            ensure_ascii=False,
        )
        now = time.time()
        pipe = self.client.pipeline(transaction=True)
        pipe.set(self._session(session_id), payload, ex=self.ttl_seconds)
        pipe.zadd(self._index(), {session_id: now})
        pipe.zremrangebyscore(self._index(), "-inf", now - self.ttl_seconds)
        pipe.execute()

    def delete(
        self, session_id: str, *, include_training_examples: bool = False
    ) -> None:
        keys = [self._session(session_id)]
        if include_training_examples:
            keys.append(self._approved(session_id))
        pipe = self.client.pipeline(transaction=True)
        pipe.delete(*keys)
        pipe.zrem(self._index(), session_id)
        pipe.execute()

    def list_sessions(self, *, limit: int = 100) -> list[dict]:
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 500
        ):
            raise ValueError("session list limit must be between 1 and 500")
        now = time.time()
        self.client.zremrangebyscore(self._index(), "-inf", now - self.ttl_seconds)
        items = self.client.zrevrange(self._index(), 0, limit - 1, withscores=True)
        out = []
        for sid, updated in items:
            raw = self.client.get(self._session(sid))
            if raw is None:
                continue
            try:
                messages = json.loads(raw)
            except Exception:
                messages = []
            user = next(
                (
                    str(x.get("content", ""))
                    for x in messages
                    if isinstance(x, dict)
                    and x.get("role") == "user"
                    and x.get("content")
                ),
                "",
            )
            assistant = next(
                (
                    str(x.get("content", ""))
                    for x in reversed(messages)
                    if isinstance(x, dict)
                    and x.get("role") == "assistant"
                    and x.get("content")
                ),
                "",
            )
            title = " ".join((user or assistant or "New conversation").split())[:96]
            out.append(
                {
                    "session_id": sid,
                    "updated": float(updated),
                    "message_count": len(messages),
                    "title": title,
                }
            )
        return out

    def review_last(self, session_id: str) -> dict | None:
        messages = [
            {"role": m.role, "content": m.content}
            for m in self.load(session_id).snapshot()
            if m.role != "system"
        ]
        for i in range(len(messages) - 1, 0, -1):
            if messages[i]["role"] == "assistant" and messages[i - 1]["role"] == "user":
                return {
                    "prompt": messages[i - 1]["content"],
                    "answer": messages[i]["content"],
                }
        return None

    def approve_last(
        self, session_id: str, *, corrected_response: str | None = None
    ) -> int:
        messages = [
            {"role": m.role, "content": m.content}
            for m in self.load(session_id).snapshot()
        ]
        if (
            not messages
            or messages[-1].get("role") != "assistant"
            or not str(messages[-1].get("content", "")).strip()
        ):
            raise ValueError("no completed assistant response to approve")
        if corrected_response is not None:
            if not corrected_response.strip():
                raise ValueError("corrected_response must be nonempty text")
            messages[-1]["content"] = corrected_response.strip()
        payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        self.client.zadd(self._approved(session_id), {payload: time.time()})
        self.client.expire(self._approved(session_id), self.ttl_seconds)
        return int(self.client.zcard(self._approved(session_id)))

    def export_training_jsonl(self, session_id: str) -> str:
        rows = self.client.zrange(self._approved(session_id), 0, -1)
        if not rows:
            raise ValueError("no approved examples to export")
        out = []
        total = 0
        for raw in rows:
            rec = json.dumps({"messages": json.loads(raw)}, ensure_ascii=False)
            total += len(rec) + 1
            if total > 1_000_000:
                raise ValueError("approved training export exceeds the 1MB limit")
            out.append(rec)
        return "\n".join(out) + "\n"

    def delete_training(self, session_id: str) -> int:
        count = int(self.client.zcard(self._approved(session_id)))
        self.client.delete(self._approved(session_id))
        return count


__all__ = ["RedisRateLimiter", "RedisIdempotencyStore", "RedisSessionStore"]
