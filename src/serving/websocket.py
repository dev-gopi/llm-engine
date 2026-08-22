"""WebSocket token-streaming endpoint."""

from __future__ import annotations

import uuid
import secrets

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError
from utils.logger import get_logger

from .runtime import ServingError
from .schemas import (
    ErrorDetail,
    FinishReason,
    GenerateRequest,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamStartEvent,
    StreamTokenEvent,
    TokenUsage,
)


logger = get_logger(__name__)
router = APIRouter()


@router.websocket("/v1/generate/stream")
async def generate_stream(websocket: WebSocket) -> None:
    settings = websocket.app.state.settings
    if settings.api_key:
        supplied = websocket.headers.get("authorization", "").removeprefix("Bearer ")
        if not secrets.compare_digest(supplied, settings.api_key):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
    origin = websocket.headers.get("origin")
    if origin and settings.cors_origins and "*" not in settings.cors_origins:
        if origin not in settings.cors_origins:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    await websocket.accept()
    runtime = websocket.app.state.runtime
    while True:
        request_id: str | None = None
        try:
            payload = await websocket.receive_json()
            request = GenerateRequest.model_validate(payload)
            request_id = f"gen_{uuid.uuid4().hex}"
            await websocket.send_json(
                StreamStartEvent(
                    id=request_id,
                    model=settings.model_name,
                    bot_name=settings.bot_name,
                ).model_dump(mode="json")
            )

            finish_reason = FinishReason.STOP
            prompt_tokens = 0
            completion_tokens = 0
            async for event in runtime.stream(request):
                prompt_tokens = event.prompt_tokens or prompt_tokens
                completion_tokens = event.completion_tokens or completion_tokens
                if event.token:
                    await websocket.send_json(
                        StreamTokenEvent(
                            id=request_id,
                            token=event.token,
                            token_id=event.token_id,
                        ).model_dump(mode="json")
                    )
                if event.finish_reason is not None:
                    finish_reason = event.finish_reason

            await websocket.send_json(
                StreamDoneEvent(
                    id=request_id,
                    finish_reason=finish_reason,
                    usage=TokenUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    ),
                ).model_dump(mode="json")
            )
        except WebSocketDisconnect:
            return
        except ValidationError as error:
            await websocket.send_json(
                StreamErrorEvent(
                    id=request_id,
                    error=ErrorDetail(
                        code="validation_error",
                        message=error.errors(include_url=False)[0]["msg"],
                    ),
                ).model_dump(mode="json")
            )
        except ServingError as error:
            await websocket.send_json(
                StreamErrorEvent(
                    id=request_id,
                    error=ErrorDetail(code=error.code, message=str(error)),
                ).model_dump(mode="json")
            )
        except Exception:
            logger.exception("unhandled streaming generation error")
            try:
                await websocket.send_json(
                    StreamErrorEvent(
                        id=request_id,
                        error=ErrorDetail(
                            code="internal_error", message="internal generation error"
                        ),
                    ).model_dump(mode="json")
                )
            except (RuntimeError, WebSocketDisconnect):
                return
