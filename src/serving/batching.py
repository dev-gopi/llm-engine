"""Async dynamic batching for backends that implement batch_generate."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass
class _WorkItem:
    request: Any
    future: asyncio.Future


class DynamicBatcher:
    def __init__(self, backend: Any, *, max_batch_size: int = 8, wait_milliseconds: float = 5.0) -> None:
        if max_batch_size < 1 or wait_milliseconds < 0:
            raise ValueError("invalid dynamic batching limits")
        if not callable(getattr(backend, "batch_generate", None)):
            raise TypeError("backend must implement async batch_generate(requests)")
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.wait_seconds = wait_milliseconds / 1000
        self.queue: asyncio.Queue[_WorkItem | None] = asyncio.Queue()
        self.worker: asyncio.Task | None = None

    async def startup(self) -> None:
        if self.worker is None:
            self.worker = asyncio.create_task(self._run())

    async def shutdown(self) -> None:
        if self.worker is not None:
            await self.queue.put(None)
            await self.worker
            self.worker = None

    async def generate(self, request: Any) -> Any:
        if self.worker is None:
            raise RuntimeError("dynamic batcher is not started")
        future = asyncio.get_running_loop().create_future()
        await self.queue.put(_WorkItem(request, future))
        return await future

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            if first is None:
                return
            batch = [first]
            if self.wait_seconds:
                await asyncio.sleep(self.wait_seconds)
            while len(batch) < self.max_batch_size:
                try:
                    item = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    await self.queue.put(None)
                    break
                batch.append(item)
            try:
                results = await self.backend.batch_generate([item.request for item in batch])
                if len(results) != len(batch):
                    raise RuntimeError("batch backend returned the wrong result count")
                for item, result in zip(batch, results, strict=True):
                    if not item.future.cancelled():
                        item.future.set_result(result)
            except Exception as error:
                for item in batch:
                    if not item.future.cancelled():
                        item.future.set_exception(error)
