"""Validated public schemas for the Gopi serving API."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, StringConstraints, field_validator


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"
    ERROR = "error"


class TextAttachment(StrictSchema):
    name: NonEmptyText = Field(max_length=255, pattern=r"^[^/\\\x00]+$")
    content: str = Field(min_length=1, max_length=262_144)
    media_type: Literal["text/plain", "text/markdown", "application/json", "text/x-code"] = "text/plain"


class GenerateRequest(StrictSchema):
    _chat_messages: list[dict[str, str]] | None = PrivateAttr(default=None)
    prompt: NonEmptyText = Field(max_length=131_072)
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]{1,128}$")
    mode: Literal["balanced", "creative", "precise", "coding"] = "balanced"
    tools: list[Literal["calculator", "datetime"]] = Field(default_factory=list, max_length=2)
    mcp: bool = False
    mcp_server: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]{1,128}$")
    response_format: Literal["plain", "markdown"] | None = None
    web_search: bool = False
    rag: bool = False
    attachments: list[TextAttachment] = Field(default_factory=list, max_length=8)
    max_tokens: int = Field(default=128, ge=1, le=8_192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, allow_inf_nan=False)
    top_k: int = Field(default=40, ge=0, le=100_000)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0, allow_inf_nan=False)
    repetition_penalty: float = Field(default=1.1, ge=0.1, le=2.0, allow_inf_nan=False)
    no_repeat_ngram_size: int = Field(default=3, ge=0, le=16)
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

    @field_validator("tools")
    @classmethod
    def deduplicate_tools(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))


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


class WorkspaceAction(StrictSchema):
    type: Literal["read", "search", "edit", "patch", "test", "git"]
    path: str = Field(default="", max_length=1024)
    query: str = Field(default="", max_length=4096)
    content: str = Field(default="", max_length=262_144)
    expected_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    apply: bool = False
    preset: Literal["unit", "all"] = "unit"
    operation: Literal["status", "diff", "diff_staged", "log"] = "status"


class WorkspaceAgentRequest(StrictSchema):
    actions: list[WorkspaceAction] = Field(min_length=1, max_length=8)


class WorkspaceAgentResponse(StrictSchema):
    results: list[dict]


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


class OpenAIChatMessage(StrictSchema):
    role: Literal["system", "user", "assistant"]
    content: NonEmptyText
    name: str | None = Field(default=None, max_length=128)


class OpenAIChatCompletionRequest(BaseModel):
    """Supported subset of the OpenAI Chat Completions request."""

    model_config = ConfigDict(extra="ignore")
    model: NonEmptyText
    messages: list[OpenAIChatMessage] = Field(min_length=1, max_length=256)
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1, le=8_192)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=8_192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, allow_inf_nan=False)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0, allow_inf_nan=False)
    stop: str | list[str] | None = None
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    user: str | None = Field(default=None, max_length=128)

    @field_validator("messages")
    @classmethod
    def require_user_message(cls, messages: list[OpenAIChatMessage]) -> list[OpenAIChatMessage]:
        if not any(message.role == "user" for message in messages):
            raise ValueError("messages must contain at least one user message")
        return messages

    def generation_request(self, expected_model: str) -> GenerateRequest:
        if self.model != expected_model:
            raise ValueError(f"unknown model: {self.model}")
        maximum = self.max_completion_tokens or self.max_tokens or 128
        stops = [self.stop] if isinstance(self.stop, str) else list(self.stop or [])
        latest_user = next(
            message.content for message in reversed(self.messages) if message.role == "user"
        )
        request = GenerateRequest(
            prompt=latest_user,
            max_tokens=maximum,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            stop=stops,
        )
        request._chat_messages = [
            {"role": message.role, "content": message.content}
            for message in self.messages
        ]
        return request


class OpenAIModel(StrictSchema):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "gopi"


class OpenAIModelList(StrictSchema):
    object: Literal["list"] = "list"
    data: list[OpenAIModel]
