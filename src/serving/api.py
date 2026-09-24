"""Production HTTP application for Gopi model serving."""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import subprocess
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Security, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.security import HTTPBearer
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from api.chat_completions import parse_response_format
from api.embeddings import EmbeddingsRequest, EmbeddingsResponse, create_embeddings
from api.responses import ResponsesOutput, ResponsesRequest, ResponsesResponse
from embeddings import EmbeddingService
from model.lifecycle import ModelLifecycleManager
from runtime.cancellation import CancellationRegistry
from runtime.capabilities import (
    discover_capabilities,
    load_capability_evidence,
    load_model_config,
)
from runtime.resource_planner import deployment_matrix, estimate_inference_memory
from utils.config import load_yaml
from utils.logger import get_logger

from .backend import backend_from_environment
from .rate_limit import InMemoryRateLimiter, SQLiteRateLimiter
from .runtime import (
    BackendGeneration,
    BackendUnavailableError,
    FinishReason,
    GenerationBackend,
    GenerationTimeoutError,
    InvalidGenerationRequestError,
    ServerBusyError,
    ServingError,
    ServingRuntime,
)
from .schemas import (
    ErrorDetail,
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    OpenAIChatCompletionRequest,
    OpenAIModel,
    OpenAIModelList,
    OpenAITool,
    SessionDeleteResponse,
    SessionListResponse,
    SessionMemoryResponse,
    TokenUsage,
    TrainingApprovalRequest,
    TrainingApprovalResponse,
    TrainingDeleteResponse,
    TrainingReviewResponse,
    WorkspaceAgentRequest,
    WorkspaceAgentResponse,
)
from .media_generation import create_media_router
from .omni import create_omni_speech_router, create_omni_video_router
from omni_platform.multimodal_input import latest_text, prepare_responses_input, synthesize_response_audio
from omni_platform.errors import OmniError
from omni_platform.speech import HuggingFaceASRProvider, HuggingFaceTTSProvider
from .websocket import router as websocket_router
from .workspace import WorkspaceService

SERVICE_NAME = "gopi-llm"
SERVICE_VERSION = "0.1.0"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
logger = get_logger(__name__)
OPENAPI_BEARER = HTTPBearer(
    auto_error=False,
    description="Optional bearer token configured through GOPI_API_KEY.",
)
OPENAPI_TAGS = [
    {
        "name": "health",
        "description": "Liveness and model-readiness checks.",
    },
    {
        "name": "generation",
        "description": "Validated text generation with optional sessions and tools.",
    },
    {
        "name": "openai-compatible",
        "description": "OpenAI Chat Completions compatibility for third-party UIs.",
    },
    {
        "name": "operations",
        "description": "Metrics and authenticated model lifecycle operations.",
    },
]


def _environment_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _add_security_headers(response: Response, *, is_https: bool) -> None:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; frame-ancestors 'none'; object-src 'none'; "
        "base-uri 'self'; img-src 'self' data:; "
        "style-src 'self'; "
        "script-src 'self'; "
        "connect-src 'self' http: https: ws: wss:"
    )
    if is_https:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"


@dataclass(frozen=True)
class ServingSettings:
    model_name: str = "gopi"
    bot_name: str = "Gopi"
    max_concurrency: int = 4
    queue_timeout_seconds: float = 1.0
    generation_timeout_seconds: float = 120.0
    cors_origins: tuple[str, ...] = ()
    api_key: str | None = None
    admin_api_key: str | None = None
    requests_per_minute: int = 0
    rate_limit_store_path: str | None = None
    continuous_streams: int = 0
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "test", "testserver")
    docs_enabled: bool = True
    protect_metrics: bool = False
    workspace_agent_enabled: bool = False
    workspace_root: str = "."
    audit_log_capacity: int = 256
    session_memory_enabled: bool = False
    embedding_model_name: str = "gopi-embedding-hash"

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if self.queue_timeout_seconds <= 0 or self.generation_timeout_seconds <= 0:
            raise ValueError("serving timeouts must be positive")
        if self.requests_per_minute < 0 or self.continuous_streams < 0:
            raise ValueError("rate and stream limits cannot be negative")
        if self.audit_log_capacity < 1:
            raise ValueError("audit_log_capacity must be positive")
        if not self.model_name.strip() or not self.bot_name.strip():
            raise ValueError("model_name and bot_name cannot be empty")
        if not self.allowed_hosts or any(not host.strip() for host in self.allowed_hosts):
            raise ValueError("allowed_hosts must contain non-empty host names")

    @classmethod
    def from_environment(cls) -> ServingSettings:
        config_path = Path(os.getenv("GOPI_INFERENCE_CONFIG", "configs/inference.yaml"))
        config = {}
        if config_path.is_file():
            config = load_yaml(config_path)
        serving = config.get("serving", {})
        if not isinstance(serving, dict):
            raise ValueError("inference serving configuration must be a mapping")
        origins = tuple(
            origin.strip()
            for origin in os.getenv(
                "GOPI_CORS_ORIGINS", ",".join(serving.get("cors_origins", []))
            ).split(",")
            if origin.strip()
        )
        allowed_hosts = tuple(
            host.strip()
            for host in os.getenv("GOPI_ALLOWED_HOSTS", "127.0.0.1,localhost,test,testserver").split(",")
            if host.strip()
        )
        return cls(
            model_name=os.getenv("GOPI_MODEL_NAME", str(serving.get("model_name", "gopi"))),
            bot_name=os.getenv("GOPI_BOT_NAME", str(config.get("bot_name", "Gopi"))),
            max_concurrency=int(
                os.getenv("GOPI_MAX_CONCURRENCY", str(serving.get("max_concurrency", 4)))
            ),
            queue_timeout_seconds=float(
                os.getenv(
                    "GOPI_QUEUE_TIMEOUT_SECONDS",
                    str(serving.get("queue_timeout_seconds", 1.0)),
                )
            ),
            generation_timeout_seconds=float(
                os.getenv(
                    "GOPI_GENERATION_TIMEOUT_SECONDS",
                    str(serving.get("generation_timeout_seconds", 120.0)),
                )
            ),
            cors_origins=origins,
            api_key=os.getenv("GOPI_API_KEY") or None,
            admin_api_key=os.getenv("GOPI_ADMIN_API_KEY") or None,
            requests_per_minute=int(os.getenv("GOPI_REQUESTS_PER_MINUTE", str(serving.get("requests_per_minute", 0)))),
            rate_limit_store_path=os.getenv("GOPI_RATE_LIMIT_STORE") or serving.get("rate_limit_store_path"),
            continuous_streams=int(os.getenv("GOPI_CONTINUOUS_STREAMS", str(serving.get("continuous_streams", 0)))),
            allowed_hosts=allowed_hosts,
            docs_enabled=_environment_flag("GOPI_DOCS_ENABLED", True),
            protect_metrics=_environment_flag("GOPI_PROTECT_METRICS", False),
            workspace_agent_enabled=_environment_flag(
                "GOPI_WORKSPACE_AGENT_ENABLED",
                bool(serving.get("workspace_agent_enabled", False)),
            ),
            workspace_root=os.getenv(
                "GOPI_WORKSPACE_ROOT", str(serving.get("workspace_root", "."))
            ),
            audit_log_capacity=int(os.getenv(
                "GOPI_AUDIT_LOG_CAPACITY", str(serving.get("audit_log_capacity", 256))
            )),
            session_memory_enabled=_environment_flag(
                "GOPI_SESSION_MEMORY_ENABLED",
                bool(serving.get("session_memory_enabled", False)),
            ),
            embedding_model_name=os.getenv("GOPI_EMBEDDING_MODEL_NAME", str(serving.get("embedding_model_name", "gopi-embedding-hash"))),
        )


class AuditLog:
    """Bounded append-only operational audit trail with no request content or secrets."""

    def __init__(self, capacity: int) -> None:
        self._events: deque[dict[str, str | int]] = deque(maxlen=capacity)

    def record(self, *, request_id: str, method: str, path: str, status_code: int) -> None:
        self._events.append({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
        })

    def events(self) -> list[dict[str, str | int]]:
        return [dict(event) for event in self._events]


def _session_context_usage(store, session_id: str, *, reserve_tokens: int) -> dict:
    """Return tokenizer-measured session capacity without exposing its content."""
    if not isinstance(reserve_tokens, int) or isinstance(reserve_tokens, bool) or reserve_tokens < 0:
        raise ValueError("reserve_tokens must be a non-negative integer")
    if reserve_tokens >= store.max_tokens:
        raise ValueError("reserve_tokens must be smaller than the context window")
    memory = store.load(session_id)
    messages = memory.snapshot()
    tokenizer = store.tokenizer

    def count(items) -> int:
        if not items:
            return 0
        text = "".join(
            f"<|{message.role}|>\n{message.content}\n<|end|>\n"
            for message in items
        )
        return len(tokenizer.encode(text, add_bos=False, allowed_special="all"))

    system = [message for message in messages if message.role == "system"]
    conversation = [message for message in messages if message.role != "system"]
    system_tokens = count(system)
    message_tokens = count(conversation)
    # These request-scoped inputs are deliberately not persisted with session
    # history. Reporting zero is both accurate and avoids retaining sensitive
    # file/tool payloads merely for telemetry.
    categories = {
        "system_instructions": system_tokens,
        "tool_definitions": 0,
        "messages": message_tokens,
        "files": 0,
        "tool_results": 0,
    }
    used_tokens = len(tokenizer.encode(
        "".join(f"<|{message.role}|>\n{message.content}\n<|end|>\n" for message in messages)
        + "<|assistant|>\n",
        add_bos=True, allowed_special="all",
    ))
    available = max(0, store.max_tokens - used_tokens - reserve_tokens)
    return {
        "session_id": session_id,
        "context_window_tokens": store.max_tokens,
        "used_tokens": used_tokens,
        "reserved_response_tokens": reserve_tokens,
        "available_tokens": available,
        "usage_percent": round(used_tokens / store.max_tokens * 100, 2),
        "categories": categories,
        "non_persisted_categories": ["tool_definitions", "files", "tool_results"],
    }


def _validate_session_id_value(session_id: str) -> None:
    if not REQUEST_ID_PATTERN.fullmatch(session_id):
        raise InvalidGenerationRequestError(
            "session id must be 1-128 characters using only letters, numbers, dot, underscore, or hyphen"
        )


async def _generate_with_disconnect(runtime: ServingRuntime, request: Request, generation_request: GenerateRequest, *, request_id: str | None = None, registry: CancellationRegistry | None = None):
    task = asyncio.create_task(runtime.generate(generation_request))
    if request_id and registry:
        registry.register(request_id, task)
    try:
        while not task.done():
            if await request.is_disconnected():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise asyncio.CancelledError("client disconnected")
            await asyncio.sleep(0.05)
        return await task
    except BaseException:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        raise
    finally:
        if request_id and registry:
            registry.unregister(request_id)


def create_app(
    backend: GenerationBackend | None = None,
    *,
    settings: ServingSettings | None = None,
) -> FastAPI:
    settings = settings or ServingSettings.from_environment()
    if settings.session_memory_enabled and not settings.api_key:
        logger.warning(
            "Server session history/training features are enabled, but GOPI_API_KEY is not configured; "
            "those endpoints will remain unavailable until authentication is configured."
        )
    if not settings.admin_api_key:
        logger.warning(
            "Model lifecycle API is unavailable because GOPI_ADMIN_API_KEY is not configured; "
            "set a dedicated admin key before using /admin/models/*."
        )
    runtime = ServingRuntime(
        backend if backend is not None else backend_from_environment(),
        max_concurrency=settings.max_concurrency,
        queue_timeout_seconds=settings.queue_timeout_seconds,
        generation_timeout_seconds=settings.generation_timeout_seconds,
        continuous_streams=settings.continuous_streams,
    )
    cancellation_registry = CancellationRegistry()
    embedding_service = EmbeddingService()
    lifecycle = ModelLifecycleManager(runtime.backend)
    def application_model_config() -> dict:
        candidate = getattr(runtime.backend, "backend", runtime.backend)
        path = getattr(candidate, "model_config", None)
        return load_model_config(path) if path is not None else {}

    application_capabilities = lambda: discover_capabilities(
        runtime.backend,
        model_config=application_model_config(),
        validation_evidence=load_capability_evidence(os.getenv("GOPI_CAPABILITY_EVIDENCE", "reports/capability_evidence.json")),
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        await runtime.startup()
        yield
        await runtime.shutdown()

    application = FastAPI(
        title="Gopi LLM API",
        summary="Local inference API for the Gopi v2 language model",
        description=(
            "Generate text with Gopi through REST or WebSocket streaming. "
            "When GOPI_API_KEY is configured, send `Authorization: Bearer <key>` "
            "to `/v1/*` REST endpoints."
        ),
        version=SERVICE_VERSION,
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
        openapi_tags=OPENAPI_TAGS,
        swagger_ui_parameters={
            "displayRequestDuration": True,
            "filter": True,
            "persistAuthorization": False,
            "tryItOutEnabled": True,
        },
    )
    application.state.runtime = runtime
    application.state.settings = settings
    workspace = (
        WorkspaceService(Path(settings.workspace_root))
        if settings.workspace_agent_enabled
        else None
    )
    application.state.workspace = workspace
    rate_limiter = (
        SQLiteRateLimiter(settings.rate_limit_store_path, settings.requests_per_minute)
        if settings.rate_limit_store_path
        else InMemoryRateLimiter(settings.requests_per_minute)
    )
    application.state.rate_limiter = rate_limiter
    audit_log = AuditLog(settings.audit_log_capacity)
    application.state.audit_log = audit_log

    application.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))

    if settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Admin-API-Key"],
            expose_headers=["X-Request-ID"],
        )

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else uuid.uuid4().hex
        request.state.request_id = request_id
        # CORS preflight requests must reach CORSMiddleware without bearer auth;
        # the actual DELETE/POST/GET request remains protected normally.
        protected_path = request.method != "OPTIONS" and (
            request.url.path.startswith("/v1/") or (
                settings.protect_metrics and request.url.path == "/metrics"
            )
        )
        if protected_path:
            if settings.api_key:
                supplied_key = request.headers.get("Authorization", "").removeprefix("Bearer ")
                if not secrets.compare_digest(supplied_key, settings.api_key):
                    response = _error_response(request, "unauthorized", "valid bearer token required", 401)
                    response.headers["X-Request-ID"] = request_id
                    _add_security_headers(response, is_https=request.url.scheme == "https")
                    audit_log.record(request_id=request_id, method=request.method,
                                     path=request.url.path, status_code=response.status_code)
                    return response
            if settings.requests_per_minute > 0:
                identity = request.client.host if request.client else "unknown"
                if not await rate_limiter.allow(identity):
                    response = _error_response(request, "rate_limit_exceeded", "request rate limit exceeded", 429)
                    response.headers["X-Request-ID"] = request_id
                    _add_security_headers(response, is_https=request.url.scheme == "https")
                    audit_log.record(request_id=request_id, method=request.method,
                                     path=request.url.path, status_code=response.status_code)
                    return response
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        _add_security_headers(response, is_https=request.url.scheme == "https")
        if protected_path:
            audit_log.record(request_id=request_id, method=request.method,
                             path=request.url.path, status_code=response.status_code)
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, error: RequestValidationError):
        first = error.errors()[0] if error.errors() else {}
        location = first.get("loc", ())
        field = ".".join(str(part) for part in location if part not in {"body", "query", "path", "header"}) or "request"
        message = str(first.get("msg", "invalid request")).removeprefix("Value error, ")
        return _error_response(request, "validation_error", f"Invalid {field}: {message}.", 422)

    @application.exception_handler(ServingError)
    async def serving_error_handler(request: Request, error: ServingError):
        status_code = {
            BackendUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
            GenerationTimeoutError: status.HTTP_504_GATEWAY_TIMEOUT,
            ServerBusyError: status.HTTP_429_TOO_MANY_REQUESTS,
            InvalidGenerationRequestError: 422,
        }.get(type(error), status.HTTP_500_INTERNAL_SERVER_ERROR)
        return _error_response(request, error.code, str(error), status_code)

    @application.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, error: Exception):
        logger.exception("unhandled serving error", exc_info=error)
        return _error_response(
            request, "internal_error", "internal server error", status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    def _require_admin(request: Request):
        """Require a dedicated admin secret; the normal API key is not enough."""
        admin_key = settings.admin_api_key
        supplied = request.headers.get("X-Admin-API-Key") or request.headers.get("Authorization", "").removeprefix("Bearer ")
        if not admin_key:
            return _error_response(
                request, "admin_auth_not_configured",
                "configure GOPI_ADMIN_API_KEY before enabling model lifecycle operations", 503,
            )
        if not supplied or not secrets.compare_digest(supplied, admin_key):
            return _error_response(request, "admin_unauthorized", "valid admin bearer token required", 403)
        return None

    def _record_admin_lifecycle(request: Request, status_code: int) -> None:
        audit_log.record(
            request_id=getattr(request.state, "request_id", "unknown"),
            method=request.method,
            path=request.url.path,
            status_code=status_code,
        )

    @application.get("/", response_model=HealthResponse, tags=["health"])
    @application.get("/health/live", response_model=HealthResponse, tags=["health"])
    async def liveness() -> HealthResponse:
        return _health(settings, runtime, status_value="ok")

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    async def health_alias() -> HealthResponse:
        return _health(settings, runtime, status_value="ok")

    @application.get(
        "/health/ready",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse}},
        tags=["health"],
    )
    async def readiness():
        payload = _health(
            settings, runtime, status_value="ready" if runtime.ready else "not_ready"
        )
        if runtime.ready:
            return payload
        return JSONResponse(status_code=503, content=payload.model_dump(mode="json"))

    @application.get("/ready", response_model=HealthResponse, responses={503: {"model": HealthResponse}}, tags=["health"])
    async def readiness_alias():
        payload = _health(settings, runtime, status_value="ready" if runtime.ready else "not_ready")
        if runtime.ready:
            return payload
        return JSONResponse(status_code=503, content=payload.model_dump(mode="json"))

    @application.get("/metrics", tags=["operations"])
    async def metrics():
        return {"service": SERVICE_NAME, "ready": runtime.ready, **runtime.metrics()}

    @application.post(
        "/v1/generate",
        response_model=GenerateResponse,
        dependencies=[Security(OPENAPI_BEARER)],
        responses={
            429: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
        },
        tags=["generation"],
    )
    async def generate(request: GenerateRequest) -> GenerateResponse:
        result = await runtime.generate(request)
        prompt_tokens = result.prompt_tokens
        completion_tokens = result.completion_tokens
        return GenerateResponse(
            id=f"gen_{uuid.uuid4().hex}",
            created=int(time.time()),
            model=settings.model_name,
            bot_name=settings.bot_name,
            text=result.text,
            finish_reason=result.finish_reason,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cached_tokens=result.cached_tokens,
                reasoning_tokens=result.reasoning_tokens,
            ),
            reasoning_content=result.reasoning_content,
            tool_calls=list(result.tool_calls) or None,
        )

    @application.get(
        "/v1/models",
        response_model=OpenAIModelList,
        tags=["openai-compatible"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def list_openai_models() -> OpenAIModelList:
        capabilities = application_capabilities()
        return OpenAIModelList(data=[OpenAIModel(
            id=settings.model_name,
            created=0,
            capabilities=capabilities.as_dict(),
            architecture=capabilities.architecture,
            context_length=capabilities.context_length,
        )])

    @application.post(
        "/v1/embeddings",
        response_model=EmbeddingsResponse,
        tags=["openai-compatible"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def embeddings(request: EmbeddingsRequest):
        if request.model != settings.embedding_model_name:
            raise InvalidGenerationRequestError("unknown embedding model")
        try:
            return create_embeddings(request, embedding_service)
        except ValueError as error:
            raise InvalidGenerationRequestError(str(error)) from error

    @application.get(
        "/v1/embeddings/models",
        tags=["openai-compatible"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def embedding_models():
        return {"data": [{
            "id": settings.embedding_model_name,
            "object": "model",
            "embedding_dimension": embedding_service.dimension,
            "dedicated": True,
            "backend": type(embedding_service.encoder).__name__,
        }]}

    @application.get(
        "/v1/models/{model_id}/capabilities",
        tags=["openai-compatible"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def model_capabilities(model_id: str, request: Request):
        if model_id != settings.model_name:
            return _error_response(request, "model_not_found", "unknown model", 404)
        return {
            "id": model_id,
            "object": "model.capabilities",
            "capabilities": application_capabilities().as_dict(),
        }

    @application.get(
        "/v1/models/{model_id}/resources",
        tags=["openai-compatible"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def model_resources(
        model_id: str, request: Request, context_length: int | None = None,
        batch_size: int = 1, weight_precision: str = "bf16",
        kv_precision: str = "bf16", memory_gib: float | None = None,
    ):
        if model_id != settings.model_name:
            return _error_response(request, "model_not_found", "unknown model", 404)
        config = application_model_config()
        if not config:
            return _error_response(request, "resource_plan_unavailable", "model configuration is unavailable", 503)
        try:
            budget = None if memory_gib is None else int(float(memory_gib) * 1024 ** 3)
            estimate = estimate_inference_memory(
                config, context_length=context_length, batch_size=batch_size,
                weight_precision=weight_precision, kv_precision=kv_precision,
                memory_budget_bytes=budget,
            )
            matrix = deployment_matrix(
                config, context_length=estimate.context_length, batch_size=batch_size,
                memory_budget_bytes=budget,
            )
        except (TypeError, ValueError) as error:
            raise InvalidGenerationRequestError(str(error)) from error
        return {
            "id": model_id,
            "object": "model.resources",
            "estimate": estimate.as_dict(),
            "alternatives": matrix,
        }

    @application.post(
        "/v1/chat/completions",
        tags=["openai-compatible"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def openai_chat_completions(request: OpenAIChatCompletionRequest, http_request: Request):
        try:
            generation_request = request.generation_request(settings.model_name)
        except ValueError as error:
            raise InvalidGenerationRequestError(str(error)) from error
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if request.stream:
            async def events():
                cancellation_registry.register(completion_id, asyncio.current_task())
                structured_parts: list[str] = []
                is_structured = isinstance(generation_request.response_format, dict) and generation_request.response_format.get("type") == "json_schema"
                start = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": settings.model_name,
                    "choices": [{
                        "index": 0, "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(start, ensure_ascii=False)}\n\n"
                finish_reason = "stop"
                async for event in runtime.stream(generation_request):
                    if await http_request.is_disconnected():
                        raise asyncio.CancelledError("client disconnected")
                    if event.reasoning_token:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": settings.model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"reasoning_content": event.reasoning_token},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    if event.token:
                        if is_structured:
                            structured_parts.append(event.token)
                            continue
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": settings.model_name,
                            "choices": [{
                                "index": 0, "delta": {"content": event.token},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    if event.tool_calls:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": settings.model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": index,
                                            **call.model_dump(mode="json"),
                                        }
                                        for index, call in enumerate(event.tool_calls)
                                    ]
                                },
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    if event.finish_reason is not None:
                        finish_reason = event.finish_reason.value
                if is_structured:
                    from schema.structured_outputs import (
                        StructuredOutputError,
                        make_spec,
                        validate_structured_output,
                    )
                    payload = generation_request.response_format["json_schema"]
                    try:
                        spec = make_spec(name=str(payload.get("name", "response")), schema=payload["schema"], strict=bool(payload.get("strict", False)))
                        validate_structured_output("".join(structured_parts).strip(), spec)
                        chunk = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": settings.model_name, "choices": [{"index": 0, "delta": {"content": "".join(structured_parts).strip()}, "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    except StructuredOutputError as exc:
                        chunk = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": settings.model_name, "choices": [{"index": 0, "delta": {"refusal": str(exc)}, "finish_reason": "length"}]}
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                done = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": settings.model_name,
                    "choices": [{
                        "index": 0, "delta": {},
                        "finish_reason": finish_reason,
                    }],
                }
                yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            async def safe_events():
                try:
                    async for event in events():
                        yield event
                finally:
                    cancellation_registry.unregister(completion_id)

            return StreamingResponse(safe_events(), media_type="text/event-stream")

        result = await _generate_with_disconnect(runtime, http_request, generation_request, request_id=completion_id, registry=cancellation_registry)
        structured_incomplete = result.structured_output_valid is False
        if not structured_incomplete:
            spec = parse_response_format(request.response_format)
            if spec is not None:
                from schema.structured_outputs import (
                    StructuredOutputError,
                    validate_structured_output,
                )
                try:
                    validate_structured_output(result.text, spec)
                except StructuredOutputError as exc:
                    structured_incomplete = True
                    result = BackendGeneration(
                        text=result.text,
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        finish_reason=FinishReason.LENGTH,
                        cached_tokens=result.cached_tokens,
                        reasoning_tokens=result.reasoning_tokens,
                        structured_output_valid=False,
                        structured_output_error=str(exc),
                        reasoning_content=result.reasoning_content,
                        tool_calls=result.tool_calls,
                        tool_call_error=result.tool_call_error,
                    )
        tool_call_incomplete = bool(result.tool_call_error and not result.tool_calls)
        incomplete_reason = None
        if structured_incomplete:
            incomplete_reason = "structured_output_validation_failed"
        elif tool_call_incomplete:
            incomplete_reason = "tool_call_generation_failed"
        message: dict[str, object | None] = {
            "role": "assistant",
            "content": (result.text or None) if result.tool_calls else result.text,
        }
        if result.reasoning_content:
            message["reasoning_content"] = result.reasoning_content
        if result.tool_calls:
            message["tool_calls"] = [call.model_dump(mode="json") for call in result.tool_calls]
        if structured_incomplete:
            message = {
                "role": "assistant",
                "content": None,
                "refusal": result.structured_output_error,
            }
        elif tool_call_incomplete:
            message["refusal"] = result.tool_call_error
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": settings.model_name,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": (
                    "length"
                    if structured_incomplete
                    else ("error" if tool_call_incomplete else result.finish_reason.value)
                ),
            }],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
                "cached_tokens": result.cached_tokens,
                "reasoning_tokens": result.reasoning_tokens,
            },
            "incomplete_details": ({"reason": incomplete_reason} if incomplete_reason else None),
        }

    @application.post(
        "/v1/responses",
        tags=["openai-compatible"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def responses(request: ResponsesRequest, http_request: Request):
        if request.model != settings.model_name:
            raise InvalidGenerationRequestError(f"unknown model: {request.model}")

        # Omni preprocesses typed image/audio/video input into the same
        # conversation representation already consumed by the native backend.
        # This preserves the existing text API and trained vision runtime.
        try:
            asr_provider = HuggingFaceASRProvider.from_env()
            tts_provider = HuggingFaceTTSProvider.from_env()
            prepared = await prepare_responses_input(
                request.input,
                asr=asr_provider,
            )
            if "audio" in request.modalities and (tts_provider is None or not tts_provider.is_available()):
                raise HTTPException(status_code=503, detail={"code": "provider_unavailable", "message": "audio output requires a configured and ready text-to-speech provider"})
        except OmniError as exc:
            raise HTTPException(status_code=exc.http_status, detail={"code": exc.code, "message": exc.message}) from exc
        messages = prepared.messages
        latest = latest_text(messages)
        if not latest:
            raise InvalidGenerationRequestError("input must contain non-empty text or supported media")

        generation_request = GenerateRequest(
            prompt=latest, max_tokens=request.max_output_tokens, temperature=request.temperature,
            top_k=request.top_k, top_p=request.top_p, min_p=request.min_p, seed=request.seed,
            stop=([request.stop] if isinstance(request.stop, str) else list(request.stop or [])),
            response_format=(request.response_format.model_dump(by_alias=True, mode="json") if request.response_format else None), reasoning_effort=request.reasoning_effort,
            session_id=request.session_id, mode=request.mode, repetition_penalty=request.repetition_penalty,
            no_repeat_ngram_size=request.no_repeat_ngram_size, min_tokens=request.min_tokens,
            rag=request.rag, web_search=request.web_search, mcp=request.mcp, mcp_server=request.mcp_server,
            chat_tools=[OpenAITool.model_validate(tool) for tool in request.tools], tool_choice=request.tool_choice,
        )
        generation_request._chat_messages = messages
        response_id = f"resp_{uuid.uuid4().hex}"
        created = int(time.time())
        if request.stream:
            async def events():
                cancellation_registry.register(response_id, asyncio.current_task())
                try:
                    yield f"data: {json.dumps({'type': 'response.created', 'response_id': response_id}, ensure_ascii=False)}\n\n"
                    for prep_event in prepared.events:
                        payload = {**prep_event, "response_id": response_id}
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    text_chunks: list[str] = []
                    async for event in runtime.stream(generation_request):
                        if await http_request.is_disconnected():
                            raise asyncio.CancelledError("client disconnected")
                        payload = {"type": "response.output_text.delta", "delta": event.token, "response_id": response_id}
                        if event.token:
                            text_chunks.append(event.token)
                        if event.reasoning_token:
                            payload = {"type": "response.reasoning.delta", "delta": event.reasoning_token, "response_id": response_id}
                        if event.tool_calls:
                            payload = {
                                "type": "response.tool_calls.delta",
                                "tool_calls": [call.model_dump(mode="json") for call in event.tool_calls],
                                "response_id": response_id,
                            }
                        if event.finish_reason is not None:
                            payload = {"type": "response.completed", "response_id": response_id, "finish_reason": event.finish_reason.value}
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    if "audio" in request.modalities and text_chunks:
                        audio = await synthesize_response_audio(
                            "".join(text_chunks), tts=tts_provider, request_id=f"{response_id}_audio"
                        )
                        yield f"data: {json.dumps({'type': 'response.audio.completed', 'response_id': response_id, 'audio': audio}, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    cancellation_registry.unregister(response_id)
            return StreamingResponse(events(), media_type="text/event-stream")
        result = await _generate_with_disconnect(runtime, http_request, generation_request, request_id=response_id, registry=cancellation_registry)
        status_value = "completed" if result.structured_output_valid is not False else "incomplete"
        output_text = result.text if status_value == "completed" else ""
        audio_output = None
        if "audio" in request.modalities and output_text:
            try:
                audio_output = await synthesize_response_audio(
                    output_text, tts=tts_provider, request_id=f"{response_id}_audio"
                )
            except OmniError as exc:
                raise HTTPException(status_code=exc.http_status, detail={"code": exc.code, "message": exc.message}) from exc
        return ResponsesResponse(
            id=response_id, created=created, model=settings.model_name, status=status_value,
            output=[ResponsesOutput(content=output_text)] if output_text else [],
            usage={"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                   "total_tokens": result.prompt_tokens + result.completion_tokens,
                   "cached_tokens": result.cached_tokens, "reasoning_tokens": result.reasoning_tokens},
            error=({"code": "structured_output_incomplete", "message": result.structured_output_error} if result.structured_output_valid is False else None),
            input_metadata=prepared.metadata or None,
            audio=audio_output,
        )

    @application.post("/v1/requests/{request_id}/cancel", tags=["operations"], dependencies=[Security(OPENAPI_BEARER)])
    async def cancel_request(request_id: str):
        if not REQUEST_ID_PATTERN.fullmatch(request_id):
            return JSONResponse(status_code=400, content={"error": {"code": "invalid_request_id", "message": "invalid request id"}})
        cancelled = cancellation_registry.cancel(request_id)
        return {"request_id": request_id, "cancelled": cancelled}

    @application.get("/admin/models", tags=["operations"], dependencies=[Security(OPENAPI_BEARER)])
    async def admin_model_status(request: Request):
        denied = _require_admin(request)
        if denied is not None:
            return denied
        state = lifecycle.state
        return {"model": settings.model_name, "status": state.status, "ready": state.ready, "version": state.version}

    @application.post("/admin/models/load", tags=["operations"], dependencies=[Security(OPENAPI_BEARER)])
    async def admin_model_load(request: Request):
        denied = _require_admin(request)
        if denied is not None:
            return denied
        try:
            state = await lifecycle.load()
        except RuntimeError as error:
            _record_admin_lifecycle(request, 409)
            return _error_response(request, "model_load_failed", str(error), 409)
        _record_admin_lifecycle(request, 200)
        return {"model": settings.model_name, "status": state.status, "ready": state.ready, "version": state.version}

    @application.post("/admin/models/unload", tags=["operations"], dependencies=[Security(OPENAPI_BEARER)])
    async def admin_model_unload(request: Request):
        denied = _require_admin(request)
        if denied is not None:
            return denied
        try:
            state = await lifecycle.unload()
        except RuntimeError as error:
            _record_admin_lifecycle(request, 409)
            return _error_response(request, "model_unload_failed", str(error), 409)
        _record_admin_lifecycle(request, 200)
        return {"model": settings.model_name, "status": state.status, "ready": state.ready, "version": state.version}

    @application.post("/admin/models/reload", tags=["operations"], dependencies=[Security(OPENAPI_BEARER)])
    async def admin_model_reload(request: Request):
        denied = _require_admin(request)
        if denied is not None:
            return denied
        try:
            state = await lifecycle.reload()
        except RuntimeError as error:
            _record_admin_lifecycle(request, 409)
            return _error_response(request, "model_reload_failed", str(error), 409)
        _record_admin_lifecycle(request, 200)
        return {"model": settings.model_name, "status": state.status, "ready": state.ready, "version": state.version}

    @application.get(
        "/v1/audit/events",
        tags=["operations"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def audit_events(request: Request):
        if not settings.api_key:
            return _error_response(
                request, "audit_disabled", "configure GOPI_API_KEY to read audit events", 403
            )
        return {"events": audit_log.events()}

    def session_store_or_error(request: Request):
        """Expose stored conversations only under an explicit admin boundary."""
        if not settings.session_memory_enabled:
            return _error_response(
                request, "session_memory_disabled",
                "enable GOPI_SESSION_MEMORY_ENABLED to access session memory", 403,
            )
        if not settings.api_key:
            return _error_response(
                request, "session_memory_auth_not_configured",
                "server conversation history is unavailable because GOPI_API_KEY is not configured; "
                "set GOPI_API_KEY and restart the server before using server-side history or training deletion.", 503,
            )
        store = getattr(runtime.backend, "sessions", None)
        if store is None:
            return _error_response(
                request, "session_memory_unavailable", "backend has no persistent session store", 409,
            )
        return store

    @application.get(
        "/v1/sessions/{session_id}/context", tags=["operations"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def session_context_info(session_id: str, request: Request, reserve_tokens: int = 128):
        store = session_store_or_error(request)
        if isinstance(store, JSONResponse):
            return store
        _validate_session_id_value(session_id)
        try:
            return _session_context_usage(store, session_id, reserve_tokens=reserve_tokens)
        except ValueError as error:
            raise InvalidGenerationRequestError(str(error)) from error

    @application.post(
        "/v1/sessions/{session_id}/context/compact", tags=["operations"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def compact_session_context(session_id: str, request: Request, reserve_tokens: int = 128):
        store = session_store_or_error(request)
        if isinstance(store, JSONResponse):
            return store
        _validate_session_id_value(session_id)
        try:
            before = _session_context_usage(store, session_id, reserve_tokens=reserve_tokens)
            memory = store.load(session_id)
            memory.render(add_generation_prompt=True, reserve_tokens=reserve_tokens)
            store.save(session_id, memory)
            after = _session_context_usage(store, session_id, reserve_tokens=reserve_tokens)
        except ValueError as error:
            raise InvalidGenerationRequestError(str(error)) from error
        return {"compacted": before["used_tokens"] != after["used_tokens"], "before": before, "after": after}

    @application.get(
        "/v1/sessions", response_model=SessionListResponse, tags=["operations"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def list_sessions(request: Request, limit: int = 100):
        store = session_store_or_error(request)
        if isinstance(store, JSONResponse):
            return store
        try:
            return {"sessions": store.list_sessions(limit=limit)}
        except ValueError as error:
            raise InvalidGenerationRequestError(str(error)) from error

    @application.get(
        "/v1/sessions/{session_id}/memory", response_model=SessionMemoryResponse, tags=["operations"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def retrieve_session_memory(session_id: str, request: Request):
        store = session_store_or_error(request)
        if isinstance(store, JSONResponse):
            return store
        _validate_session_id_value(session_id)
        messages = [
            {"role": message.role, "content": message.content}
            for message in store.load(session_id).snapshot()
        ]
        return {"session_id": session_id, "messages": messages}

    @application.delete(
        "/v1/sessions/{session_id}/memory", response_model=SessionDeleteResponse, tags=["operations"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def delete_session_memory(session_id: str, request: Request):
        store = session_store_or_error(request)
        if isinstance(store, JSONResponse):
            return store
        _validate_session_id_value(session_id)
        store.delete(session_id, include_training_examples=True)
        return {"session_id": session_id, "deleted": True}

    @application.get(
        "/v1/sessions/{session_id}/training/review", response_model=TrainingReviewResponse, tags=["operations"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def review_session_training(session_id: str, request: Request):
        store = session_store_or_error(request)
        if isinstance(store, JSONResponse):
            return store
        _validate_session_id_value(session_id)
        review = store.review_last(session_id)
        if review is None:
            raise InvalidGenerationRequestError("no completed assistant response is available for review")
        return {"session_id": session_id, **review}

    @application.post(
        "/v1/sessions/{session_id}/training/approve", response_model=TrainingApprovalResponse, tags=["operations"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def approve_session_training(session_id: str, payload: TrainingApprovalRequest, request: Request):
        store = session_store_or_error(request)
        if isinstance(store, JSONResponse):
            return store
        _validate_session_id_value(session_id)
        if not payload.approved:
            return {"session_id": session_id, "approved": False, "example_count": 0}
        try:
            count = store.approve_last(session_id, corrected_response=payload.corrected_response)
        except ValueError as error:
            raise InvalidGenerationRequestError(str(error)) from error
        return {"session_id": session_id, "approved": True, "example_count": count}

    @application.get(
        "/v1/sessions/{session_id}/training/export", tags=["operations"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def export_session_training(session_id: str, request: Request):
        store = session_store_or_error(request)
        if isinstance(store, JSONResponse):
            return store
        _validate_session_id_value(session_id)
        try:
            content = store.export_training_jsonl(session_id)
        except ValueError as error:
            raise InvalidGenerationRequestError(str(error)) from error
        return Response(
            content=content, media_type="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="reviewed-chat-{session_id}.jsonl"'},
        )

    @application.delete(
        "/v1/sessions/{session_id}/training", response_model=TrainingDeleteResponse, tags=["operations"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def delete_session_training(session_id: str, request: Request):
        store = session_store_or_error(request)
        if isinstance(store, JSONResponse):
            return store
        _validate_session_id_value(session_id)
        return {"session_id": session_id, "deleted_count": store.delete_training(session_id)}

    @application.post(
        "/v1/workspace/actions",
        response_model=WorkspaceAgentResponse,
        tags=["operations"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def workspace_actions(
        payload: WorkspaceAgentRequest, request: Request
    ) -> WorkspaceAgentResponse | JSONResponse:
        if workspace is None:
            return _error_response(
                request,
                "workspace_agent_disabled",
                "enable GOPI_WORKSPACE_AGENT_ENABLED to use workspace actions",
                403,
            )
        if not settings.api_key:
            return _error_response(
                request,
                "workspace_auth_required",
                "configure GOPI_API_KEY before enabling workspace actions",
                403,
            )
        try:
            results = await run_in_threadpool(workspace.execute, payload.actions)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            raise InvalidGenerationRequestError(str(error)) from error
        return WorkspaceAgentResponse(results=results)

    ui_directory = Path(__file__).resolve().parents[2] / "ui"
    favicon_path = ui_directory / "favicon.ico"

    @application.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        return FileResponse(
        favicon_path,
        media_type="image/x-icon",
        filename="favicon.ico",
        headers={"Cache-Control": "public, max-age=86400"},
    )

    if ui_directory.is_dir():
        ui_assets = {
            "index": (ui_directory / "index.html").read_text(encoding="utf-8"),
            "styles": (ui_directory / "styles.css").read_text(encoding="utf-8"),
            "script": (ui_directory / "app.js").read_text(encoding="utf-8"),
        }

        @application.get("/ui", include_in_schema=False)
        @application.get("/ui/", include_in_schema=False)
        async def playground() -> HTMLResponse:
            return HTMLResponse(ui_assets["index"])

        @application.get("/ui/styles.css", include_in_schema=False)
        async def playground_styles() -> Response:
            return Response(ui_assets["styles"], media_type="text/css")

        @application.get("/ui/app.js", include_in_schema=False)
        async def playground_script() -> Response:
            return Response(ui_assets["script"], media_type="text/javascript")

    application.include_router(create_media_router())
    application.include_router(create_omni_speech_router())
    application.include_router(create_omni_video_router(runtime))
    application.include_router(websocket_router)
    return application


def _health(
    settings: ServingSettings,
    runtime: ServingRuntime,
    *,
    status_value: str,
) -> HealthResponse:
    return HealthResponse(
        status=status_value,
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        model=settings.model_name,
        ready=runtime.ready,
        authentication_required=bool(settings.api_key),
    )


def _error_response(
    request: Request, code: str, message: str, status_code: int
) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=getattr(request.state, "request_id", None),
        )
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


app = create_app()
