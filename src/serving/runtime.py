"""Concurrency, lifecycle, and backend isolation for model serving."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from utils.logger import get_logger

from .orchestration import ContinuousStreamScheduler, TokenStepScheduler
from .schemas import FinishReason, GenerateRequest, OpenAIToolCall

logger = get_logger(__name__)

if TYPE_CHECKING:
    from .semantic_cache import SemanticResponseCache


class ServingError(RuntimeError):
    code = "serving_error"


class BackendUnavailableError(ServingError):
    code = "backend_unavailable"


class GenerationTimeoutError(ServingError):
    code = "generation_timeout"


class ServerBusyError(ServingError):
    code = "server_busy"


class InvalidGenerationRequestError(ServingError):
    code = "invalid_generation_request"


@dataclass(frozen=True)
class BackendGeneration:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: FinishReason = FinishReason.STOP
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    structured_output_valid: bool | None = None
    structured_output_error: str | None = None
    reasoning_content: str | None = None
    tool_calls: tuple[OpenAIToolCall, ...] = ()
    tool_call_error: str | None = None
    logprobs: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class BackendStreamEvent:
    token: str = ""
    token_id: int | None = None
    finish_reason: FinishReason | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_token: str = ""
    tool_calls: tuple[OpenAIToolCall, ...] = ()
    tool_call_error: str | None = None
    logprob: dict[str, object] | None = None


@runtime_checkable
class GenerationBackend(Protocol):
    @property
    def ready(self) -> bool: ...

    async def generate(self, request: GenerateRequest) -> BackendGeneration: ...

    def stream(self, request: GenerateRequest) -> AsyncIterator[BackendStreamEvent]: ...


class UnavailableBackend:
    """Safe default used until model loading is wired into the application."""

    ready = False

    def _production_cache_context(self, request: GenerateRequest):
        if self.production_semantic_cache is None:
            return None
        from .production_semantic_cache import ProductionCacheRequest
        from .semantic_cache_policy import CacheFreshness

        freshness = (
            CacheFreshness(request.cache_fingerprint)
            if request.cache_fingerprint
            else None
        )
        return ProductionCacheRequest(
            tenant=request.tenant_id,
            user=request.user_id,
            route=request.route,
            model=request.model_id,
            task=request.task,
            freshness=freshness,
        )

    async def _production_cached_generate(
        self, request: GenerateRequest
    ) -> BackendGeneration:
        cache = self.production_semantic_cache
        context = self._production_cache_context(request)
        eligible, _reason = cache.policy(request, context)
        if not eligible:
            return await self.backend.generate(request)
        key = cache.key(request, tenant=context.tenant, freshness=context.freshness)
        negative = cache.negative_get(key)
        if negative is not None:
            raise ServingError(f"cached backend error: {negative.error_type}")
        cached, _similarity = await cache.lookup(request, context, key)
        if cached is not None:
            return cached

        async def work():
            result = await self.backend.generate(request)
            await cache.store(request, context, key, result)
            return result

        try:
            return await cache.singleflight(key, context.tenant, work)
        except Exception as exc:
            cache.negative_put(key, type(exc).__name__, str(exc))
            raise

    async def generate(self, request: GenerateRequest) -> BackendGeneration:
        raise BackendUnavailableError("generation backend is not loaded")

    async def stream(
        self, request: GenerateRequest
    ) -> AsyncIterator[BackendStreamEvent]:
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
        continuous_streams: int = 0,
        metrics_window: int = 256,
        semantic_cache: SemanticResponseCache | None = None,
        production_semantic_cache: object | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if queue_timeout_seconds <= 0 or generation_timeout_seconds <= 0:
            raise ValueError("serving timeouts must be positive")
        if metrics_window < 1:
            raise ValueError("metrics_window must be positive")
        self.backend = backend or UnavailableBackend()
        self.semantic_cache = semantic_cache
        self.production_semantic_cache = production_semantic_cache
        self.max_concurrency = max_concurrency
        self.queue_timeout_seconds = queue_timeout_seconds
        self.generation_timeout_seconds = generation_timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)
        scheduler_backend = backend or self.backend
        if continuous_streams and all(
            callable(getattr(scheduler_backend, method, None))
            for method in ("start_stream", "decode_stream_batch")
        ):
            self.stream_scheduler = TokenStepScheduler(
                scheduler_backend, max_active=continuous_streams
            )
            self.stream_scheduler_mode = "token_step"
        elif continuous_streams:
            self.stream_scheduler = ContinuousStreamScheduler(
                scheduler_backend, max_active=continuous_streams
            )
            self.stream_scheduler_mode = "independent"
        else:
            self.stream_scheduler = None
            self.stream_scheduler_mode = "disabled"
        self.active_requests = 0
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.total_generation_seconds = 0.0
        self.total_queue_seconds = 0.0
        self.total_completion_tokens = 0
        self._latencies: deque[float] = deque(maxlen=metrics_window)
        self._ttft: deque[float] = deque(maxlen=metrics_window)

    @property
    def ready(self) -> bool:
        try:
            return bool(self.backend.ready)
        except Exception:
            return False

    async def startup(self) -> None:
        await self._call_lifecycle("startup")
        if (
            self.production_semantic_cache is not None
            and getattr(self.production_semantic_cache, "redis", None) is not None
        ):
            await self.production_semantic_cache.redis.connect()
        if self.stream_scheduler:
            await self.stream_scheduler.startup()

    async def shutdown(self) -> None:
        if self.stream_scheduler:
            await self.stream_scheduler.shutdown()
        await self._call_lifecycle("shutdown")
        if self.semantic_cache is not None:
            self.semantic_cache.close()
        if (
            self.production_semantic_cache is not None
            and getattr(self.production_semantic_cache, "redis", None) is not None
        ):
            client = getattr(self.production_semantic_cache.redis, "client", None)
            if client is not None:
                close = getattr(client, "aclose", None) or getattr(
                    client, "close", None
                )
                if close is not None:
                    result = close()
                    if inspect.isawaitable(result):
                        await result

    def _production_cache_context(self, request: GenerateRequest):
        if self.production_semantic_cache is None:
            return None
        from .production_semantic_cache import ProductionCacheRequest
        from .semantic_cache_policy import CacheFreshness

        freshness = (
            CacheFreshness(request.cache_fingerprint)
            if request.cache_fingerprint
            else None
        )
        return ProductionCacheRequest(
            tenant=request.tenant_id,
            user=request.user_id,
            route=request.route,
            model=request.model_id,
            task=request.task,
            freshness=freshness,
        )

    async def _production_cached_generate(
        self, request: GenerateRequest
    ) -> BackendGeneration:
        cache = self.production_semantic_cache
        context = self._production_cache_context(request)
        eligible, _reason = cache.policy(request, context)
        if not eligible:
            return await self.backend.generate(request)
        key = cache.key(request, tenant=context.tenant, freshness=context.freshness)
        negative = cache.negative_get(key)
        if negative is not None:
            raise ServingError(f"cached backend error: {negative.error_type}")
        cached, _similarity = await cache.lookup(request, context, key)
        if cached is not None:
            return cached

        async def work():
            result = await self.backend.generate(request)
            await cache.store(request, context, key, result)
            return result

        try:
            return await cache.singleflight(key, context.tenant, work)
        except Exception as exc:
            cache.negative_put(key, type(exc).__name__, str(exc))
            raise

    async def generate(self, request: GenerateRequest) -> BackendGeneration:
        if not self.ready:
            raise BackendUnavailableError("generation backend is not ready")
        await self._acquire()
        started = time.monotonic()
        try:
            async with asyncio.timeout(self.generation_timeout_seconds):
                if self.production_semantic_cache is not None:
                    result = await self._production_cached_generate(request)
                elif self.semantic_cache is not None and self.semantic_cache.eligible(
                    request
                ):
                    async with self.semantic_cache.lock_for(request):
                        cached = self.semantic_cache.lookup(request)
                        if cached is not None:
                            result = cached
                        else:
                            result = await self.backend.generate(request)
                            self.semantic_cache.store(request, result)
                else:
                    if self.semantic_cache is not None:
                        self.semantic_cache.lookup(request)  # records a bypass
                    result = await self.backend.generate(request)
                self.completed_requests += 1
                self.total_completion_tokens += result.completion_tokens
                return result
        except TimeoutError as error:
            self.failed_requests += 1
            raise GenerationTimeoutError("generation exceeded its deadline") from error
        except Exception:
            self.failed_requests += 1
            raise
        finally:
            elapsed = time.monotonic() - started
            self.total_generation_seconds += elapsed
            self._latencies.append(elapsed)
            self._release()

    async def stream(
        self, request: GenerateRequest
    ) -> AsyncIterator[BackendStreamEvent]:
        if not self.ready:
            raise BackendUnavailableError("generation backend is not ready")
        await self._acquire()
        started = time.monotonic()
        first_token_at: float | None = None
        completion_tokens = 0
        try:
            async with asyncio.timeout(self.generation_timeout_seconds):
                protocol_sensitive = bool(
                    request.reasoning_effort != "none"
                    or (request.chat_tools and request.tool_choice != "none")
                    or any(
                        isinstance(message.get("content"), list)
                        for message in (request._chat_messages or [])
                    )
                )
                cached = None
                cache_lock = None
                production_context = None
                production_key = None
                if self.production_semantic_cache is not None:
                    production_context = self._production_cache_context(request)
                    eligible, _reason = self.production_semantic_cache.policy(
                        request, production_context
                    )
                    if eligible:
                        production_key = self.production_semantic_cache.key(
                            request,
                            tenant=production_context.tenant,
                            freshness=production_context.freshness,
                        )
                        (
                            cached,
                            _similarity,
                        ) = await self.production_semantic_cache.lookup(
                            request, production_context, production_key
                        )
                elif self.semantic_cache is not None and self.semantic_cache.eligible(
                    request
                ):
                    cache_lock = self.semantic_cache.lock_for(request)
                    await cache_lock.acquire()
                    cached = self.semantic_cache.lookup(request)
                elif self.semantic_cache is not None:
                    self.semantic_cache.lookup(request)  # records a bypass
                try:
                    if cached is not None:
                        first_token_at = time.monotonic()
                        completion_tokens = cached.completion_tokens
                        if cached.text:
                            yield BackendStreamEvent(
                                token=cached.text,
                                prompt_tokens=cached.prompt_tokens,
                                completion_tokens=cached.completion_tokens,
                            )
                        yield BackendStreamEvent(
                            finish_reason=cached.finish_reason,
                            prompt_tokens=cached.prompt_tokens,
                            completion_tokens=cached.completion_tokens,
                        )
                    else:
                        source = (
                            self.stream_scheduler.stream(request)
                            if self.stream_scheduler and not protocol_sensitive
                            else self.backend.stream(request)
                        )
                        pieces: list[str] = []
                        prompt_tokens = 0
                        finish_reason = FinishReason.STOP
                        async for event in source:
                            if first_token_at is None and (
                                event.token or event.reasoning_token or event.tool_calls
                            ):
                                first_token_at = time.monotonic()
                            if event.token:
                                pieces.append(event.token)
                            prompt_tokens = max(prompt_tokens, event.prompt_tokens)
                            completion_tokens = max(
                                completion_tokens, event.completion_tokens
                            )
                            if event.finish_reason is not None:
                                finish_reason = event.finish_reason
                            yield event
                        generated = BackendGeneration(
                            text="".join(pieces),
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            finish_reason=finish_reason,
                        )
                        if (
                            production_key is not None
                            and production_context is not None
                        ):
                            await self.production_semantic_cache.store(
                                request, production_context, production_key, generated
                            )
                        elif (
                            self.semantic_cache is not None
                            and self.semantic_cache.eligible(request)
                        ):
                            self.semantic_cache.store(request, generated)
                finally:
                    if cache_lock is not None and cache_lock.locked():
                        cache_lock.release()
            self.completed_requests += 1
            self.total_completion_tokens += completion_tokens
        except TimeoutError as error:
            self.failed_requests += 1
            raise GenerationTimeoutError(
                "streaming generation exceeded its deadline"
            ) from error
        except Exception:
            self.failed_requests += 1
            raise
        finally:
            elapsed = time.monotonic() - started
            self.total_generation_seconds += elapsed
            self._latencies.append(elapsed)
            if first_token_at is not None:
                self._ttft.append(first_token_at - started)
            self._release()

    def metrics(self) -> dict[str, int | float | str]:
        elapsed = max(self.total_generation_seconds, 1e-9)
        metrics: dict[str, int | float | str] = {
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "total_generation_seconds": self.total_generation_seconds,
            "total_queue_seconds": self.total_queue_seconds,
            "tokens_per_second": self.total_completion_tokens / elapsed,
            "error_rate": self.failed_requests / max(self.total_requests, 1),
            "latency_p50_seconds": self._percentile(self._latencies, 0.50),
            "latency_p95_seconds": self._percentile(self._latencies, 0.95),
            "ttft_p50_seconds": self._percentile(self._ttft, 0.50),
            "stream_scheduler_mode": self.stream_scheduler_mode,
        }
        allocator = getattr(
            getattr(self.backend, "generator", None), "paged_kv_allocator", None
        )
        if allocator is not None:
            capacity = len(allocator.free_pages) + sum(
                len(table) for table in allocator.tables.values()
            )
            metrics["paged_kv_pages_used"] = capacity - len(allocator.free_pages)
            metrics["paged_kv_page_utilization"] = (
                metrics["paged_kv_pages_used"] / capacity if capacity else 0.0
            )
        generator = getattr(self.backend, "generator", None)
        if generator is not None:
            from evaluation.prefix_cache import collect_prefix_cache_metrics

            metrics.update(
                {
                    f"prefix_cache_{key}": value
                    for key, value in collect_prefix_cache_metrics(generator)
                    .to_dict()
                    .items()
                }
            )
        if self.semantic_cache is not None:
            metrics.update(
                {
                    f"semantic_cache_{key}": value
                    for key, value in self.semantic_cache.metrics().items()
                }
            )
        if self.production_semantic_cache is not None:
            metrics.update(
                {
                    f"semantic_cache_distributed_{key}": value
                    for key, value in self.production_semantic_cache.metrics().items()
                }
            )
        return metrics

    async def _acquire(self) -> None:
        queued = time.monotonic()
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self.queue_timeout_seconds
            )
        except TimeoutError as error:
            raise ServerBusyError("all generation workers are busy") from error
        self.active_requests += 1
        self.total_requests += 1
        self.total_queue_seconds += time.monotonic() - queued

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

    @staticmethod
    def _percentile(values: deque[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = round((len(ordered) - 1) * percentile)
        return ordered[index]
