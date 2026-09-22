"""Protected model lifecycle orchestration.

The lifecycle manager deliberately exposes only configured backend factories;
it never accepts arbitrary filesystem paths from an HTTP request.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LifecycleState:
    status: str
    version: str | None
    ready: bool


class ModelLifecycleManager:
    def __init__(self, backend: Any) -> None:
        self.backend = backend

    @property
    def state(self) -> LifecycleState:
        return LifecycleState(
            status="loaded" if bool(getattr(self.backend, "ready", False)) else "unloaded",
            version=getattr(self.backend, "version", None),
            ready=bool(getattr(self.backend, "ready", False)),
        )

    async def load(self) -> LifecycleState:
        callback = getattr(self.backend, "load_current", None)
        if callback is not None:
            await callback()
        elif not getattr(self.backend, "ready", False):
            callback = getattr(self.backend, "startup", None)
            if callback is None:
                raise RuntimeError("backend does not support loading")
            result = callback()
            if hasattr(result, "__await__"):
                await result
        if not getattr(self.backend, "ready", False):
            raise RuntimeError("model did not become ready after load")
        return self.state

    async def unload(self) -> LifecycleState:
        callback = getattr(self.backend, "unload", None)
        if callback is None:
            raise RuntimeError("backend does not support unloading")
        await callback()
        return self.state

    async def reload(self) -> LifecycleState:
        callback = getattr(self.backend, "reload_current", None)
        if callback is None:
            raise RuntimeError("backend does not support reloading")
        await callback()
        return self.state
