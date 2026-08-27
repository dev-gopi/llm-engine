"""Process-local and SQLite-backed fixed-window request rate limiting."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections import defaultdict, deque
from pathlib import Path


class InMemoryRateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.limit = requests_per_minute
        self.windows: dict[str, deque[float]] = defaultdict(deque)
        self.lock = asyncio.Lock()

    async def allow(self, identity: str) -> bool:
        if self.limit <= 0:
            return True
        async with self.lock:
            now = time.time()
            window = self.windows[identity]
            while window and window[0] <= now - 60:
                window.popleft()
            if len(window) >= self.limit:
                return False
            window.append(now)
            return True


class SQLiteRateLimiter:
    """Cross-process limiter using an atomic SQLite immediate transaction."""

    def __init__(self, path: str | Path, requests_per_minute: int) -> None:
        self.path = Path(path)
        self.limit = requests_per_minute
        self.lock = asyncio.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS rate_events "
                "(identity TEXT NOT NULL, occurred REAL NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS rate_events_lookup "
                "ON rate_events(identity, occurred)"
            )

    async def allow(self, identity: str) -> bool:
        if self.limit <= 0:
            return True
        # The transaction is deliberately tiny. Keeping it on the event-loop
        # thread avoids creating non-daemon executor threads during application
        # shutdown while SQLite still serializes independent worker processes.
        async with self.lock:
            return self._allow_sync(identity, time.time())

    def _allow_sync(self, identity: str, now: float) -> bool:
        with sqlite3.connect(self.path, timeout=5, isolation_level=None) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM rate_events WHERE occurred <= ?", (now - 60,))
            count = connection.execute(
                "SELECT COUNT(*) FROM rate_events WHERE identity = ?", (identity,)
            ).fetchone()[0]
            if count >= self.limit:
                connection.execute("COMMIT")
                return False
            connection.execute(
                "INSERT INTO rate_events(identity, occurred) VALUES (?, ?)", (identity, now)
            )
            connection.execute("COMMIT")
            return True
