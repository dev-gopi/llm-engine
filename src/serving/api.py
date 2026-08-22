"""Production HTTP application for Gopi model serving."""

from __future__ import annotations

import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from utils.config import load_yaml
from utils.logger import get_logger

from .runtime import (
    BackendUnavailableError,
    GenerationBackend,
    GenerationTimeoutError,
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
    TokenUsage,
)
from .websocket import router as websocket_router


SERVICE_NAME = "gopi-llm"
SERVICE_VERSION = "0.1.0"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
logger = get_logger(__name__)


@dataclass(frozen=True)
class ServingSettings:
    model_name: str = "gopi"
    bot_name: str = "Gopi"
    max_concurrency: int = 4
    queue_timeout_seconds: float = 1.0
    generation_timeout_seconds: float = 120.0
    cors_origins: tuple[str, ...] = ()

    @classmethod
    def from_environment(cls) -> "ServingSettings":
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
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        await runtime.startup()
        yield
        await runtime.shutdown()

    application = FastAPI(
        title="Gopi LLM API",
        version=SERVICE_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    application.state.runtime = runtime
    application.state.settings = settings

    if settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID"],
        )

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
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

    @application.post(
        "/v1/generate",
        response_model=GenerateResponse,
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
