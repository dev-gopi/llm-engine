"""OpenAI-compatible Responses API request/response contracts."""
from __future__ import annotations
from typing import Any, Literal
from serving.schemas import OpenAITool
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    top_k: int = Field(default=40, ge=0)
    min_p: float = Field(default=0, ge=0, le=1)
    seed: int | None = Field(default=None, ge=0)
    stop: str | list[str] | None = None
    response_format: dict[str, Any] | None = None
    reasoning_effort: Literal["none", "low", "medium", "high"] = "none"
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: Any = "auto"

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
