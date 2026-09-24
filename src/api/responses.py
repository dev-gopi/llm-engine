"""OpenAI-compatible Responses API request/response contracts.

Omni v7 keeps the original string/message forms while adding typed multimodal
input parts. Media parts reference either content-safe image URLs/data URLs or
assets previously uploaded through the media asset API.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from serving.schemas import OpenAIResponseFormat, OpenAITool, OpenAIToolChoice


class ResponseInputText(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["input_text"] = "input_text"
    text: str = Field(min_length=1, max_length=262_144)


class ResponseInputImage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["input_image"] = "input_image"
    image_url: str | None = Field(default=None, max_length=16_777_216)
    asset_id: str | None = Field(default=None, pattern=r"^asset_[0-9a-f]{32}$")
    detail: Literal["auto", "low", "high"] = "auto"

    @model_validator(mode="after")
    def exactly_one_source(self):
        if bool(self.image_url) == bool(self.asset_id):
            raise ValueError("input_image requires exactly one of image_url or asset_id")
        return self


class ResponseInputAudio(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["input_audio"] = "input_audio"
    asset_id: str = Field(pattern=r"^asset_[0-9a-f]{32}$")
    language: str | None = Field(default=None, max_length=32)


class ResponseInputVideo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["input_video"] = "input_video"
    asset_id: str = Field(pattern=r"^asset_[0-9a-f]{32}$")
    max_frames: int | None = Field(default=None, ge=1, le=32)
    language: str | None = Field(default=None, max_length=32)


ResponseContentPart = Annotated[
    ResponseInputText | ResponseInputImage | ResponseInputAudio | ResponseInputVideo,
    Field(discriminator="type"),
]


class ResponseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str | list[ResponseContentPart]

    @field_validator("content")
    @classmethod
    def validate_content(cls, value):
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("message content cannot be empty")
            if len(value) > 262_144:
                raise ValueError("message content cannot exceed 262144 characters")
        elif not value:
            raise ValueError("message content parts cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_role_media(self):
        if isinstance(self.content, list) and self.role != "user":
            if any(not isinstance(part, ResponseInputText) for part in self.content):
                raise ValueError("image/audio/video input parts are only accepted on user messages")
        return self


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1, max_length=128)
    input: str | list[ResponseInput]
    stream: bool = False
    modalities: list[Literal["text", "audio"]] = Field(default_factory=lambda: ["text"], min_length=1, max_length=2)
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


    @field_validator("modalities")
    @classmethod
    def validate_modalities(cls, values):
        normalized = list(dict.fromkeys(values))
        if "text" not in normalized:
            raise ValueError("Omni v7 Responses currently requires text output; audio may be added alongside text")
        return normalized

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
    input_metadata: dict[str, Any] | None = None
    audio: dict[str, Any] | None = None
