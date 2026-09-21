"""Bounded tool execution with cancellation and timeout propagation."""
from __future__ import annotations
import asyncio
from typing import Awaitable, Callable, Any

async def execute_tool(tool: Callable[..., Awaitable[Any]], *args, timeout: float = 30.0, cancellation: asyncio.Event | None = None, **kwargs) -> Any:
    task = asyncio.create_task(tool(*args, **kwargs))
    try:
        if cancellation is None: return await asyncio.wait_for(task, timeout=timeout)
        cancel_wait = asyncio.create_task(cancellation.wait())
        done, pending = await asyncio.wait({task, cancel_wait}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if cancel_wait in done and cancellation.is_set():
            task.cancel(); await asyncio.gather(task, return_exceptions=True); raise asyncio.CancelledError
        cancel_wait.cancel()
        await asyncio.gather(cancel_wait, return_exceptions=True)
        if task not in done: task.cancel(); await asyncio.gather(task, return_exceptions=True); raise TimeoutError("tool execution timed out")
        return task.result()
    except BaseException:
        if not task.done(): task.cancel(); await asyncio.gather(task, return_exceptions=True)
        raise
