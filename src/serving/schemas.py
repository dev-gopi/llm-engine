"""Validated public schemas for the Gopi serving API."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"
    ERROR = "error"


class GenerateRequest(StrictSchema):
    prompt: NonEmptyText = Field(max_length=131_072)
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]{1,128}$")
    max_tokens: int = Field(default=512, ge=1, le=8_192)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, allow_inf_nan=False)
    top_k: int = Field(default=40, ge=0, le=100_000)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0, allow_inf_nan=False)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=2.0, allow_inf_nan=False)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    stop: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("stop")
    @classmethod
    def validate_stop_sequences(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            if not value:
                raise ValueError("stop sequences cannot be empty")
            if len(value) > 1_024:
                raise ValueError("stop sequences cannot exceed 1024 characters")
            if value not in normalized:
                normalized.append(value)
        return normalized


class TokenUsage(StrictSchema):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class GenerateResponse(StrictSchema):
    id: str
    object: Literal["text_generation"] = "text_generation"
    created: int
    model: str
    bot_name: str
    text: str
    finish_reason: FinishReason
    usage: TokenUsage


class HealthResponse(StrictSchema):
    status: Literal["ok", "ready", "not_ready"]
    service: str
    version: str
    model: str
    ready: bool


class ErrorDetail(StrictSchema):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(StrictSchema):
    error: ErrorDetail


class StreamStartEvent(StrictSchema):
    type: Literal["start"] = "start"
    id: str
    model: str
    bot_name: str


class StreamTokenEvent(StrictSchema):
    type: Literal["token"] = "token"
    id: str
    token: str
    token_id: int | None = None


class StreamDoneEvent(StrictSchema):
    type: Literal["done"] = "done"
    id: str
    finish_reason: FinishReason
    usage: TokenUsage


class StreamErrorEvent(StrictSchema):
    type: Literal["error"] = "error"
    id: str | None = None
    error: ErrorDetail
