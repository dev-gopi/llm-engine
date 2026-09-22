"""Process-local and SQLite-backed fixed-window request rate limiting."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RateLimitDecision:
    """Structured limiter result for HTTP headers and observability."""

    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: float


class InMemoryRateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.limit = requests_per_minute
        self.windows: dict[str, deque[float]] = defaultdict(deque)
        self.lock = asyncio.Lock()
        self._operations = 0

    async def allow(self, identity: str) -> bool:
        return (await self.check(identity)).allowed

    async def check(self, identity: str) -> RateLimitDecision:
        if self.limit <= 0:
            return RateLimitDecision(True, self.limit, self.limit, 0.0)
        async with self.lock:
            now = time.monotonic()
            self._operations += 1
            if self._operations % 1024 == 0:
                self._prune(now)
            window = self.windows[identity]
            cutoff = now - 60.0
            while window and window[0] <= cutoff:
                window.popleft()
            if len(window) >= self.limit:
                retry_after = max(0.0, 60.0 - (now - window[0]))
                return RateLimitDecision(False, self.limit, 0, retry_after)
            window.append(now)
            return RateLimitDecision(
                True, self.limit, max(self.limit - len(window), 0), 0.0
            )

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        for key, events in list(self.windows.items()):
            while events and events[0] <= cutoff:
                events.popleft()
            if not events:
                self.windows.pop(key, None)


class SQLiteRateLimiter:
    """Cross-process limiter using an atomic SQLite immediate transaction."""

    def __init__(self, path: str | Path, requests_per_minute: int) -> None:
        self.path = Path(path)
        self.limit = requests_per_minute
        self.lock = asyncio.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=5) as connection:
            # WAL permits readers while another worker commits a small rate-limit
            # transaction and significantly reduces avoidable lock contention.
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS rate_events "
                "(identity TEXT NOT NULL, occurred REAL NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS rate_events_lookup "
                "ON rate_events(identity, occurred)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS rate_events_expiry ON rate_events(occurred)"
            )

    async def allow(self, identity: str) -> bool:
        return (await self.check(identity)).allowed

    async def check(self, identity: str) -> RateLimitDecision:
        if self.limit <= 0:
            return RateLimitDecision(True, self.limit, self.limit, 0.0)
        async with self.lock:
            return self._check_sync(identity, time.time())

    def _allow_sync(self, identity: str, now: float) -> bool:
        """Backward-compatible internal hook retained for callers/tests."""
        return self._check_sync(identity, now).allowed

    def _check_sync(self, identity: str, now: float) -> RateLimitDecision:
        with sqlite3.connect(self.path, timeout=5, isolation_level=None) as connection:
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("BEGIN IMMEDIATE")
            try:
                cutoff = now - 60.0
                connection.execute("DELETE FROM rate_events WHERE occurred <= ?", (cutoff,))
                row = connection.execute(
                    "SELECT COUNT(*), MIN(occurred) FROM rate_events WHERE identity = ?",
                    (identity,),
                ).fetchone()
                count = int(row[0])
                oldest = row[1]
                if count >= self.limit:
                    retry_after = (
                        max(0.0, 60.0 - (now - float(oldest)))
                        if oldest is not None
                        else 60.0
                    )
                    connection.execute("COMMIT")
                    return RateLimitDecision(False, self.limit, 0, retry_after)
                connection.execute(
                    "INSERT INTO rate_events(identity, occurred) VALUES (?, ?)",
                    (identity, now),
                )
                connection.execute("COMMIT")
                return RateLimitDecision(
                    True, self.limit, max(self.limit - count - 1, 0), 0.0
                )
            except BaseException:
                connection.execute("ROLLBACK")
                raise
