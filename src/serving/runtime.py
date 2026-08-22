"""Concurrency, lifecycle, and backend isolation for model serving."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .schemas import FinishReason, GenerateRequest


class ServingError(RuntimeError):
    code = "serving_error"


class BackendUnavailableError(ServingError):
    code = "backend_unavailable"


class GenerationTimeoutError(ServingError):
    code = "generation_timeout"


class ServerBusyError(ServingError):
    code = "server_busy"


@dataclass(frozen=True)
class BackendGeneration:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: FinishReason = FinishReason.STOP


@dataclass(frozen=True)
class BackendStreamEvent:
    token: str = ""
    token_id: int | None = None
    finish_reason: FinishReason | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@runtime_checkable
class GenerationBackend(Protocol):
    @property
    def ready(self) -> bool: ...

    async def generate(self, request: GenerateRequest) -> BackendGeneration: ...

    def stream(self, request: GenerateRequest) -> AsyncIterator[BackendStreamEvent]: ...


class UnavailableBackend:
    """Safe default used until model loading is wired into the application."""

    ready = False

    async def generate(self, request: GenerateRequest) -> BackendGeneration:
        raise BackendUnavailableError("generation backend is not loaded")

    async def stream(self, request: GenerateRequest) -> AsyncIterator[BackendStreamEvent]:
        raise BackendUnavailableError("generation backend is not loaded")
        yield  # pragma: no cover


class ServingRuntime:
    def __init__(
        self,
        backend: GenerationBackend | None = None,
        *,
        max_concurrency: int = 4,
        queue_timeout_seconds: float = 1.0,
        generation_timeout_seconds: float = 120.0,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if queue_timeout_seconds <= 0 or generation_timeout_seconds <= 0:
            raise ValueError("serving timeouts must be positive")
        self.backend = backend or UnavailableBackend()
        self.max_concurrency = max_concurrency
        self.queue_timeout_seconds = queue_timeout_seconds
        self.generation_timeout_seconds = generation_timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.active_requests = 0
        self.total_requests = 0

    @property
    def ready(self) -> bool:
        try:
            return bool(self.backend.ready)
        except Exception:
            return False

    async def startup(self) -> None:
        await self._call_lifecycle("startup")

    async def shutdown(self) -> None:
        await self._call_lifecycle("shutdown")

    async def generate(self, request: GenerateRequest) -> BackendGeneration:
        if not self.ready:
            raise BackendUnavailableError("generation backend is not ready")
        await self._acquire()
        try:
            async with asyncio.timeout(self.generation_timeout_seconds):
                return await self.backend.generate(request)
        except TimeoutError as error:
            raise GenerationTimeoutError("generation exceeded its deadline") from error
        finally:
            self._release()

    async def stream(self, request: GenerateRequest) -> AsyncIterator[BackendStreamEvent]:
        if not self.ready:
            raise BackendUnavailableError("generation backend is not ready")
        await self._acquire()
        try:
            async with asyncio.timeout(self.generation_timeout_seconds):
                async for event in self.backend.stream(request):
                    yield event
        except TimeoutError as error:
            raise GenerationTimeoutError("streaming generation exceeded its deadline") from error
        finally:
            self._release()

    async def _acquire(self) -> None:
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self.queue_timeout_seconds
            )
        except TimeoutError as error:
            raise ServerBusyError("all generation workers are busy") from error
        self.active_requests += 1
        self.total_requests += 1

    def _release(self) -> None:
        self.active_requests -= 1
        self._semaphore.release()

    async def _call_lifecycle(self, method_name: str) -> None:
        method = getattr(self.backend, method_name, None)
        if method is None:
            return
        result = method()
        if inspect.isawaitable(result):
            await result
