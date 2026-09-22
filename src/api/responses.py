"""OpenAI-compatible Responses API request/response contracts."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from serving.schemas import OpenAIResponseFormat, OpenAITool, OpenAIToolChoice


class ResponseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str = Field(min_length=1, max_length=262144)

class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1, max_length=128)
    input: str | list[ResponseInput]
    stream: bool = False
    max_output_tokens: int = Field(default=128, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int = Field(default=40, ge=0, le=100_000)
    min_p: float = Field(default=0, ge=0, le=1)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    stop: str | list[str] | None = None
    response_format: OpenAIResponseFormat | None = None
    reasoning_effort: Literal["none", "low", "medium", "high"] = "none"
    tools: list[dict[str, Any]] = Field(default_factory=list, max_length=128)
    tool_choice: OpenAIToolChoice = "auto"
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]{1,128}$")
    mode: Literal["balanced", "creative", "precise", "coding"] = "balanced"
    repetition_penalty: float = Field(default=1.1, ge=0.1, le=2.0)
    no_repeat_ngram_size: int = Field(default=3, ge=0, le=16)
    min_tokens: int = Field(default=1, ge=0, le=1_000_000)
    mcp: bool = False
    mcp_server: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]{1,128}$")

    @field_validator("input")
    @classmethod
    def validate_input(cls, value):
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("input cannot be empty")
            if len(value) > 262_144:
                raise ValueError("input cannot exceed 262144 characters")
            return value
        if not value:
            raise ValueError("input must contain at least one message")
        if len(value) > 256:
            raise ValueError("input cannot contain more than 256 messages")
        return value

    @field_validator("stop")
    @classmethod
    def validate_stop(cls, value):
        values = [value] if isinstance(value, str) else list(value or [])
        normalized = []
        for item in values:
            if not item:
                raise ValueError("stop sequences cannot be empty")
            if len(item) > 1024:
                raise ValueError("stop sequences cannot exceed 1024 characters")
            if item not in normalized:
                normalized.append(item)
        return normalized[0] if isinstance(value, str) else normalized if value is not None else None

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, values):
        return [OpenAITool.model_validate(value).model_dump(mode="json") for value in values]

    rag: bool = False
    web_search: bool = False

class ResponsesOutput(BaseModel):
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: str

class ResponsesResponse(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created: int
    model: str
    status: Literal["completed", "incomplete", "failed"]
    output: list[ResponsesOutput]
    usage: dict[str, Any]
    error: dict[str, Any] | None = None
