"""Production HTTP application for Gopi model serving."""

from __future__ import annotations

import os
import json
import re
import time
import uuid
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Request, Security, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPBearer
from utils.config import load_yaml
from utils.logger import get_logger

from .runtime import (
    BackendUnavailableError,
    GenerationBackend,
    GenerationTimeoutError,
    InvalidGenerationRequestError,
    ServerBusyError,
    ServingError,
    ServingRuntime,
)
from .backend import backend_from_environment
from .schemas import (
    ErrorDetail,
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    OpenAIChatCompletionRequest,
    OpenAIModel,
    OpenAIModelList,
    TokenUsage,
)
from .websocket import router as websocket_router
from .rate_limit import InMemoryRateLimiter, SQLiteRateLimiter


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
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "connect-src 'self' ws: wss:"
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
    requests_per_minute: int = 0
    rate_limit_store_path: str | None = None
    continuous_streams: int = 0
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "test", "testserver")
    docs_enabled: bool = True
    protect_metrics: bool = False

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if self.queue_timeout_seconds <= 0 or self.generation_timeout_seconds <= 0:
            raise ValueError("serving timeouts must be positive")
        if self.requests_per_minute < 0 or self.continuous_streams < 0:
            raise ValueError("rate and stream limits cannot be negative")
        if not self.model_name.strip() or not self.bot_name.strip():
            raise ValueError("model_name and bot_name cannot be empty")
        if not self.allowed_hosts or any(not host.strip() for host in self.allowed_hosts):
            raise ValueError("allowed_hosts must contain non-empty host names")

    @classmethod
    def from_environment(cls) -> "ServingSettings":
        config_path = Path(os.getenv("GOPI_INFERENCE_CONFIG", "configs/inference.v2.yaml"))
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
            requests_per_minute=int(os.getenv("GOPI_REQUESTS_PER_MINUTE", str(serving.get("requests_per_minute", 0)))),
            rate_limit_store_path=os.getenv("GOPI_RATE_LIMIT_STORE") or serving.get("rate_limit_store_path"),
            continuous_streams=int(os.getenv("GOPI_CONTINUOUS_STREAMS", str(serving.get("continuous_streams", 0)))),
            allowed_hosts=allowed_hosts,
            docs_enabled=_environment_flag("GOPI_DOCS_ENABLED", True),
            protect_metrics=_environment_flag("GOPI_PROTECT_METRICS", False),
        )


def create_app(
    backend: GenerationBackend | None = None,
    *,
    settings: ServingSettings | None = None,
) -> FastAPI:
    settings = settings or ServingSettings.from_environment()
    runtime = ServingRuntime(
        backend if backend is not None else backend_from_environment(),
        max_concurrency=settings.max_concurrency,
        queue_timeout_seconds=settings.queue_timeout_seconds,
        generation_timeout_seconds=settings.generation_timeout_seconds,
        continuous_streams=settings.continuous_streams,
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
    rate_limiter = (
        SQLiteRateLimiter(settings.rate_limit_store_path, settings.requests_per_minute)
        if settings.rate_limit_store_path
        else InMemoryRateLimiter(settings.requests_per_minute)
    )
    application.state.rate_limiter = rate_limiter

    application.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))

    if settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID"],
        )

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else uuid.uuid4().hex
        request.state.request_id = request_id
        protected_path = request.url.path.startswith("/v1/") or (
            settings.protect_metrics and request.url.path == "/metrics"
        )
        if protected_path:
            if settings.api_key:
                supplied_key = request.headers.get("Authorization", "").removeprefix("Bearer ")
                if not secrets.compare_digest(supplied_key, settings.api_key):
                    response = _error_response(request, "unauthorized", "valid bearer token required", 401)
                    response.headers["X-Request-ID"] = request_id
                    _add_security_headers(response, is_https=request.url.scheme == "https")
                    return response
            if settings.requests_per_minute > 0:
                identity = request.client.host if request.client else "unknown"
                if not await rate_limiter.allow(identity):
                    response = _error_response(request, "rate_limit_exceeded", "request rate limit exceeded", 429)
                    response.headers["X-Request-ID"] = request_id
                    _add_security_headers(response, is_https=request.url.scheme == "https")
                    return response
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        _add_security_headers(response, is_https=request.url.scheme == "https")
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, error: RequestValidationError):
        message = error.errors()[0].get("msg", "invalid request")
        return _error_response(request, "validation_error", message, 422)

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

    @application.get("/", response_model=HealthResponse, tags=["health"])
    @application.get("/health/live", response_model=HealthResponse, tags=["health"])
    async def liveness() -> HealthResponse:
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
            ),
        )

    @application.get(
        "/v1/models",
        response_model=OpenAIModelList,
        tags=["openai-compatible"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def list_openai_models() -> OpenAIModelList:
        return OpenAIModelList(data=[OpenAIModel(
            id=settings.model_name,
            created=0,
        )])

    @application.post(
        "/v1/chat/completions",
        tags=["openai-compatible"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def openai_chat_completions(request: OpenAIChatCompletionRequest):
        try:
            generation_request = request.generation_request(settings.model_name)
        except ValueError as error:
            raise InvalidGenerationRequestError(str(error)) from error
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if request.stream:
            async def events():
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
                    if event.token:
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
                    if event.finish_reason is not None:
                        finish_reason = event.finish_reason.value
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

            return StreamingResponse(events(), media_type="text/event-stream")

        result = await runtime.generate(generation_request)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": settings.model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": result.finish_reason.value,
            }],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        }

    @application.post(
        "/v1/admin/reload",
        tags=["operations"],
        dependencies=[Security(OPENAPI_BEARER)],
    )
    async def reload_model(request: Request):
        if not settings.api_key:
            return _error_response(
                request, "reload_disabled", "configure GOPI_API_KEY to enable model reload", 403
            )
        callback = getattr(runtime.backend, "reload_current", None)
        if callback is None:
            return _error_response(
                request, "reload_unavailable", "backend does not support reload", 409
            )
        version = await callback()
        return {"status": "reloaded", "version": version}

    ui_directory = Path(__file__).resolve().parents[2] / "ui"
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
