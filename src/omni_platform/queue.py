"""Queue abstraction with local development and optional Redis production backends."""
from __future__ import annotations

import json
import queue
import time
from dataclasses import asdict, dataclass
from typing import Protocol

from .errors import DependencyMissingError


@dataclass(slots=True)
class QueueMessage:
    id: str
    payload: dict
    priority: int = 0
    attempts: int = 0
    available_at: float = 0.0


class JobQueue(Protocol):
    def put(self, message: QueueMessage) -> None: ...
    def get(self, *, timeout: float = 1.0) -> QueueMessage | None: ...
    def dead_letter(self, message: QueueMessage, reason: str) -> None: ...


class InProcessJobQueue:
    """Bounded priority queue appropriate for local development and tests."""

    def __init__(self, maxsize: int = 64) -> None:
        self._queue: queue.PriorityQueue[tuple[int, int, QueueMessage]] = queue.PriorityQueue(maxsize=maxsize)
        self._seq = 0
        self.dead_letters: list[tuple[QueueMessage, str]] = []

    def put(self, message: QueueMessage) -> None:
        self._seq += 1
        # Higher numeric priority executes first.
        self._queue.put((-message.priority, self._seq, message), block=False)

    def get(self, *, timeout: float = 1.0) -> QueueMessage | None:
        deadline = time.time() + timeout
        while True:
            remaining = max(0.0, deadline - time.time())
            try:
                _, _, msg = self._queue.get(timeout=remaining)
            except queue.Empty:
                return None
            if msg.available_at <= time.time():
                return msg
            self.put(msg)
            if time.time() >= deadline:
                return None
            time.sleep(min(0.05, max(0.0, msg.available_at - time.time())))

    def dead_letter(self, message: QueueMessage, reason: str) -> None:
        self.dead_letters.append((message, reason))


class RedisJobQueue:
    """Redis sorted-set queue with priority, delayed availability, and dead-letter storage."""

    def __init__(self, url: str, *, name: str = "gopi:media") -> None:
        try:
            import redis  # type: ignore
        except ImportError as exc:
            raise DependencyMissingError("install redis to use RedisJobQueue") from exc
        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.name = name
        self.ready_key = f"{name}:ready"
        self.payload_key = f"{name}:payload"
        self.dead_key = f"{name}:dead"

    def put(self, message: QueueMessage) -> None:
        now = time.time()
        available = message.available_at or now
        # Score sorts first by availability; priority is encoded as a small subtraction.
        score = available - max(-1000, min(1000, message.priority)) * 1e-6
        blob = json.dumps(asdict(message), separators=(",", ":"), sort_keys=True)
        pipe = self.client.pipeline()
        pipe.hset(self.payload_key, message.id, blob)
        pipe.zadd(self.ready_key, {message.id: score})
        pipe.execute()

    def get(self, *, timeout: float = 1.0) -> QueueMessage | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            ids = self.client.zrangebyscore(self.ready_key, "-inf", time.time(), start=0, num=1)
            if ids:
                message_id = ids[0]
                # Remove atomically enough for a single Redis instance; workers race on zrem result.
                if self.client.zrem(self.ready_key, message_id) == 1:
                    raw = self.client.hget(self.payload_key, message_id)
                    self.client.hdel(self.payload_key, message_id)
                    if raw:
                        return QueueMessage(**json.loads(raw))
            time.sleep(0.05)
        return None

    def dead_letter(self, message: QueueMessage, reason: str) -> None:
        self.client.rpush(self.dead_key, json.dumps({"message": asdict(message), "reason": reason}, sort_keys=True))
