"""Token-stream scheduling, replica routing, and zero-downtime backend reloads."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class _StreamWork:
    request: Any
    queue: asyncio.Queue


class ContinuousStreamScheduler:
    """Multiplex active backend streams and admit new requests continuously."""

    def __init__(self, backend: Any, *, max_active: int = 8, queue_size: int = 32) -> None:
        if max_active < 1 or queue_size < 1:
            raise ValueError("continuous scheduler limits must be positive")
        self.backend = backend
        self.max_active = max_active
        self.pending: asyncio.Queue[_StreamWork | None] = asyncio.Queue(maxsize=queue_size)
        self.worker: asyncio.Task | None = None

    async def startup(self) -> None:
        if self.worker is None:
            self.worker = asyncio.create_task(self._run())

    async def shutdown(self) -> None:
        if self.worker is not None:
            await self.pending.put(None)
            await self.worker
            self.worker = None

    async def stream(self, request: Any) -> AsyncIterator[Any]:
        if self.worker is None:
            raise RuntimeError("continuous scheduler is not started")
        queue: asyncio.Queue = asyncio.Queue()
        await self.pending.put(_StreamWork(request, queue))
        while True:
            item = await queue.get()
            if isinstance(item, _StreamEnd):
                if item.error is not None:
                    raise item.error
                return
            yield item

    async def _run(self) -> None:
        active: set[asyncio.Task] = set()
        stopping = False
        intake: asyncio.Task | None = None
        while not stopping or active:
            if not stopping and len(active) < self.max_active and intake is None:
                intake = asyncio.create_task(self.pending.get())
            waiters = set(active)
            if intake is not None:
                waiters.add(intake)
            if not waiters:
                break
            done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            if intake is not None and intake in done:
                work = intake.result()
                intake = None
                if work is None:
                    stopping = True
                else:
                    active.add(asyncio.create_task(self._pump(work)))
            active.difference_update(task for task in done if task is not intake)

    async def _pump(self, work: _StreamWork) -> None:
        try:
            async for event in self.backend.stream(work.request):
                await work.queue.put(event)
        except Exception as error:
            await work.queue.put(_StreamEnd(error))
        else:
            await work.queue.put(_StreamEnd())


@dataclass(frozen=True)
class _StreamEnd:
    error: Exception | None = None


class ReplicaPoolBackend:
    """Least-active routing across compatible generation backends."""

    def __init__(self, replicas: Sequence[Any]) -> None:
        if not replicas:
            raise ValueError("replica pool cannot be empty")
        self.replicas = list(replicas)
        self.active = [0] * len(replicas)
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return all(bool(replica.ready) for replica in self.replicas)

    async def startup(self) -> None:
        await asyncio.gather(*(self._lifecycle(replica, "startup") for replica in self.replicas))

    async def shutdown(self) -> None:
        await asyncio.gather(*(self._lifecycle(replica, "shutdown") for replica in self.replicas))

    async def generate(self, request):
        index, replica = await self._acquire()
        try:
            return await replica.generate(request)
        finally:
            await self._release(index)

    async def stream(self, request):
        index, replica = await self._acquire()
        try:
            async for event in replica.stream(request):
                yield event
        finally:
            await self._release(index)

    async def _acquire(self):
        async with self._lock:
            ready = [index for index, replica in enumerate(self.replicas) if replica.ready]
            if not ready:
                raise RuntimeError("no model replica is ready")
            index = min(ready, key=lambda candidate: (self.active[candidate], candidate))
            self.active[index] += 1
            return index, self.replicas[index]

    async def _release(self, index: int) -> None:
        async with self._lock:
            self.active[index] -= 1

    @staticmethod
    async def _lifecycle(replica, method: str) -> None:
        callback = getattr(replica, method, None)
        if callback is not None:
            result = callback()
            if asyncio.iscoroutine(result):
                await result


class ReloadableBackend:
    """Atomically swap a warmed backend and drain the previous generation."""

    def __init__(self, backend: Any, *, version: str = "initial", factory=None) -> None:
        self.backend = backend
        self.version = version
        self.factory = factory
        self._active: dict[int, int] = {id(backend): 0}
        self._condition = asyncio.Condition()

    @property
    def ready(self) -> bool:
        return bool(self.backend.ready)

    async def startup(self) -> None:
        callback = getattr(self.backend, "startup", None)
        if callback:
            await callback()

    async def shutdown(self) -> None:
        callback = getattr(self.backend, "shutdown", None)
        if callback:
            await callback()

    async def reload(self, backend: Any, *, version: str) -> None:
        callback = getattr(backend, "startup", None)
        if callback:
            await callback()
        if not backend.ready:
            shutdown = getattr(backend, "shutdown", None)
            if shutdown:
                await shutdown()
            raise RuntimeError("replacement backend did not become ready")
        async with self._condition:
            previous = self.backend
            self.backend = backend
            self.version = version
            self._active.setdefault(id(backend), 0)
            while self._active.get(id(previous), 0):
                await self._condition.wait()
            self._active.pop(id(previous), None)
        shutdown = getattr(previous, "shutdown", None)
        if shutdown:
            await shutdown()

    async def reload_current(self) -> str:
        if self.factory is None:
            raise RuntimeError("no reload factory is configured")
        backend, version = self.factory()
        await self.reload(backend, version=version)
        return version

    async def generate(self, request):
        backend = await self._acquire()
        try:
            return await backend.generate(request)
        finally:
            await self._release(backend)

    async def stream(self, request):
        backend = await self._acquire()
        try:
            async for event in backend.stream(request):
                yield event
        finally:
            await self._release(backend)

    async def _acquire(self):
        async with self._condition:
            backend = self.backend
            self._active[id(backend)] = self._active.get(id(backend), 0) + 1
            return backend

    async def _release(self, backend) -> None:
        async with self._condition:
            self._active[id(backend)] -= 1
            self._condition.notify_all()
