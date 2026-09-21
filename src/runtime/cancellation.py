"""Request-wide cancellation token."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field

@dataclass
class CancellationToken:
    event: asyncio.Event = field(default_factory=asyncio.Event)
    reason: str | None = None
    def cancel(self, reason: str = "client_disconnect") -> None:
        self.reason = reason; self.event.set()
    @property
    def cancelled(self) -> bool: return self.event.is_set()
    def raise_if_cancelled(self) -> None:
        if self.cancelled: raise asyncio.CancelledError(self.reason or "cancelled")

class CancellationRegistry:
    """Bounded in-process registry for explicit request cancellation."""
    def __init__(self, capacity: int = 1024):
        if capacity < 1: raise ValueError("capacity must be positive")
        self._tasks: dict[str, asyncio.Task] = {}
        self._capacity = capacity
    def register(self, request_id: str, task: asyncio.Task) -> None:
        if len(self._tasks) >= self._capacity:
            stale = next(iter(self._tasks))
            self._tasks.pop(stale, None)
        self._tasks[request_id] = task
    def unregister(self, request_id: str) -> None: self._tasks.pop(request_id, None)
    def cancel(self, request_id: str) -> bool:
        task = self._tasks.get(request_id)
        if task is None: return False
        task.cancel("explicit client cancellation")
        return True
    def active(self, request_id: str) -> bool: return request_id in self._tasks
