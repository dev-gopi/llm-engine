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
    task: asyncio.Task | None = None
    cancelled: bool = False


class ContinuousStreamScheduler:
    """Multiplex active backend streams and admit new requests continuously."""

    def __init__(
        self, backend: Any, *, max_active: int = 8, queue_size: int = 32,
        event_queue_size: int = 64,
    ) -> None:
        if max_active < 1 or queue_size < 1 or event_queue_size < 1:
            raise ValueError("continuous scheduler limits must be positive")
        self.backend = backend
        self.max_active = max_active
        self.pending: asyncio.Queue[_StreamWork | None] = asyncio.Queue(maxsize=queue_size)
        self.event_queue_size = event_queue_size
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
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.event_queue_size)
        work = _StreamWork(request, queue)
        await self.pending.put(work)
        try:
            while True:
                item = await queue.get()
                if isinstance(item, _StreamEnd):
                    if item.error is not None:
                        raise item.error
                    return
                yield item
        finally:
            work.cancelled = True
            if work.task is not None and not work.task.done():
                work.task.cancel()

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
                elif not work.cancelled:
                    work.task = asyncio.create_task(self._pump(work))
                    active.add(work.task)
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


@dataclass
class _TokenWork:
    request: Any
    queue: asyncio.Queue
    state: Any = None
    cancelled: bool = False


class TokenStepScheduler:
    """Continuously admit requests and execute one batched decode step per tick.

    The backend must implement ``start_stream(request)``,
    ``decode_stream_batch(states)`` returning one ``(event, done)`` pair per
    state, and may implement ``release_stream(state)`` for KV-page reclamation.
    """

    def __init__(self, backend: Any, *, max_active: int = 32, queue_size: int = 1024) -> None:
        for method in ("start_stream", "decode_stream_batch"):
            if not callable(getattr(backend, method, None)):
                raise TypeError(f"token-step backend must implement {method}()")
        if max_active < 1 or queue_size < 1:
            raise ValueError("token scheduler limits must be positive")
        self.backend, self.max_active = backend, max_active
        self.pending: asyncio.Queue[_TokenWork | None] = asyncio.Queue(maxsize=queue_size)
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
            raise RuntimeError("token scheduler is not started")
        work = _TokenWork(request, asyncio.Queue(maxsize=2))
        await self.pending.put(work)
        try:
            while True:
                value = await work.queue.get()
                if isinstance(value, _StreamEnd):
                    if value.error:
                        raise value.error
                    return
                yield value
        finally:
            work.cancelled = True

    async def _run(self) -> None:
        active: list[_TokenWork] = []
        stopping = False
        while active or not stopping:
            while not stopping and len(active) < self.max_active:
                try:
                    item = self.pending.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    stopping = True
                    break
                if not item.cancelled:
                    try:
                        item.state = await self.backend.start_stream(item.request)
                        active.append(item)
                    except Exception as error:
                        await item.queue.put(_StreamEnd(error))
            if not active:
                if stopping:
                    break
                item = await self.pending.get()
                if item is None:
                    stopping = True
                else:
                    try:
                        item.state = await self.backend.start_stream(item.request)
                        active.append(item)
                    except Exception as error:
                        await item.queue.put(_StreamEnd(error))
                continue
            cancelled = [item for item in active if item.cancelled]
            release = getattr(self.backend, "release_stream", None)
            for item in cancelled:
                if release:
                    result = release(item.state)
                    if asyncio.iscoroutine(result):
                        await result
            live = [item for item in active if not item.cancelled]
            if not live:
                active = []
                continue
            results = await self.backend.decode_stream_batch([item.state for item in live])
            if len(results) != len(live):
                raise RuntimeError("token-step backend returned the wrong result count")
            survivors = []
            for item, (event, done) in zip(live, results, strict=True):
                if event is not None:
                    await item.queue.put(event)
                if done or item.cancelled:
                    if release:
                        result = release(item.state)
                        if asyncio.iscoroutine(result):
                            await result
                    await item.queue.put(_StreamEnd())
                else:
                    survivors.append(item)
            active = survivors
            await asyncio.sleep(0)


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
