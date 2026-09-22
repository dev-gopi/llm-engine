"""Validated public schemas for the Gopi serving API."""

from __future__ import annotations

from enum import Enum
import json
from typing import Annotated, Any, Literal

from jsonschema import exceptions as jsonschema_exceptions
from jsonschema.validators import validator_for

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StringConstraints,
    field_validator,
    model_validator,
)

from schema.structured_outputs import make_spec, StructuredSchemaError

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"
    ERROR = "error"
    TOOL_CALLS = "tool_calls"


class TextAttachment(StrictSchema):
    name: NonEmptyText = Field(max_length=255, pattern=r"^[^/\\\x00]+$")
    content: str = Field(min_length=1, max_length=262_144)
    media_type: Literal[
        "text/plain",
        "text/markdown",
        "application/json",
        "text/x-code",
    ] = "text/plain"


class GenerateRequest(StrictSchema):
    _chat_messages: list[dict[str, Any]] | None = PrivateAttr(default=None)

    prompt: NonEmptyText = Field(max_length=131_072)
    session_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9._-]{1,128}$",
    )
    mode: Literal["balanced", "creative", "precise", "coding"] = "balanced"

    # Legacy/internal tool names.
    tools: list[str] = Field(default_factory=list, max_length=128)

    mcp: bool = False
    mcp_server: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9._-]{1,128}$",
    )

    response_format: str | dict[str, Any] | None = None
    reasoning_effort: Literal["none", "low", "medium", "high"] = "none"
    web_search: bool = False
    rag: bool = False

    attachments: list[TextAttachment] = Field(
        default_factory=list,
        max_length=8,
    )

    # Keep the public API free of a small hard-coded ceiling. The loaded model
    # context window is the authoritative runtime limit and is enforced by the
    # backend before generation starts.
    max_tokens: int = Field(default=128, ge=1, le=1_000_000)
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        allow_inf_nan=False,
    )
    top_k: int = Field(default=40, ge=0, le=100_000)
    top_p: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    min_p: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    repetition_penalty: float = Field(
        default=1.1,
        ge=0.1,
        le=2.0,
        allow_inf_nan=False,
    )
    no_repeat_ngram_size: int = Field(default=3, ge=0, le=16)
    min_tokens: int = Field(default=1, ge=0, le=1_000_000)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)

    stop: list[str] = Field(default_factory=list, max_length=16)

    # OpenAI-compatible function/tool definitions.
    chat_tools: list["OpenAITool"] = Field(
        default_factory=list,
        max_length=128,
    )
    tool_choice: "OpenAIToolChoice" = "auto"

    @field_validator("response_format")
    @classmethod
    def validate_response_format(cls, value):
        if value is None or isinstance(value, dict):
            return value
        if value not in {"plain", "markdown"}:
            raise ValueError("response format must be plain, markdown, or a structured response object")
        return value

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
        normalized = list(dict.fromkeys(values))
        unsupported = set(normalized) - {"calculator", "datetime"}
        if unsupported:
            raise ValueError(
                f"unsupported legacy tools: {', '.join(sorted(unsupported))}"
            )
        return normalized

    @field_validator("chat_tools")
    @classmethod
    def validate_chat_tools(cls, values: list["OpenAITool"]) -> list["OpenAITool"]:
        names: set[str] = set()

        for tool in values:
            name = tool.function.name

            if name in names:
                raise ValueError(f"duplicate tool name: {name}")

            names.add(name)

        return values


class TokenUsage(StrictSchema):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)


class OpenAIToolFunction(StrictSchema):
    name: NonEmptyText = Field(
        max_length=128,
        pattern=r"^[A-Za-z_][A-Za-z0-9_-]*$",
    )
    description: str | None = Field(default=None, max_length=16_384)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        schema = value or {"type": "object", "properties": {}}
        try:
            validator_for(schema).check_schema(schema)
        except jsonschema_exceptions.SchemaError as exc:
            raise ValueError(f"invalid function parameter schema: {exc.message}") from exc
        if schema.get("type") not in {None, "object"}:
            raise ValueError("function parameters must describe a JSON object")
        return value


class OpenAITool(StrictSchema):
    type: Literal["function"] = "function"
    function: OpenAIToolFunction


class OpenAIToolFunctionCall(StrictSchema):
    name: NonEmptyText = Field(max_length=128)
    arguments: str = Field(default="{}", max_length=262_144)

    @field_validator("arguments")
    @classmethod
    def validate_arguments(cls, value: str) -> str:
        try:
            parsed = json.loads(value)
        except ValueError as exc:
            raise ValueError("tool call arguments must be valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("tool call arguments must encode a JSON object")
        return value


class OpenAIToolCall(StrictSchema):
    id: NonEmptyText = Field(max_length=128)
    type: Literal["function"] = "function"
    function: OpenAIToolFunctionCall


class OpenAIToolChoiceFunction(StrictSchema):
    name: NonEmptyText = Field(max_length=128)


class OpenAIToolChoiceObject(StrictSchema):
    type: Literal["function"] = "function"
    function: OpenAIToolChoiceFunction


OpenAIToolChoice = (
    Literal["none", "auto", "required"]
    | OpenAIToolChoiceObject
)


class OpenAIChatTextPart(StrictSchema):
    type: Literal["text"] = "text"
    text: str = Field(min_length=1, max_length=262_144)


class OpenAIImageURL(StrictSchema):
    url: NonEmptyText = Field(max_length=16_777_216)
    detail: Literal["auto", "low", "high"] = "auto"


class OpenAIChatImagePart(StrictSchema):
    type: Literal["image_url"] = "image_url"
    image_url: OpenAIImageURL


OpenAIChatContentPart = Annotated[
    OpenAIChatTextPart | OpenAIChatImagePart,
    Field(discriminator="type"),
]


class OpenAIChatMessage(BaseModel):
    """OpenAI-compatible chat message.

    `content` may be null for an assistant message containing tool_calls.
    """

    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[OpenAIChatContentPart] | None = None
    name: str | None = Field(default=None, max_length=128)

    # Assistant -> requested tool execution.
    tool_calls: list[OpenAIToolCall] | None = Field(
        default=None,
        max_length=128,
    )

    # Tool -> result returned to the model.
    tool_call_id: str | None = Field(
        default=None,
        max_length=128,
    )

    @field_validator("content")
    @classmethod
    def validate_content(
        cls,
        value: str | list[OpenAIChatContentPart] | None,
    ) -> str | list[OpenAIChatContentPart] | None:
        if isinstance(value, str) and not value.strip():
            raise ValueError("message content cannot be empty")
        if isinstance(value, list) and not value:
            raise ValueError("message content parts cannot be empty")

        return value

    def text_content(self, *, image_placeholder: bool = True) -> str:
        """Return a text fallback suitable for safety checks and native prompts."""
        if isinstance(self.content, str):
            return self.content
        if not self.content:
            return ""
        parts: list[str] = []
        for part in self.content:
            if isinstance(part, OpenAIChatTextPart):
                parts.append(part.text)
            elif image_placeholder:
                parts.append("[image]")
        return "\n".join(part for part in parts if part).strip()

    def image_urls(self) -> list[str]:
        if not isinstance(self.content, list):
            return []
        return [
            part.image_url.url
            for part in self.content
            if isinstance(part, OpenAIChatImagePart)
        ]

    @model_validator(mode="after")
    def validate_role_payload(self):
        if isinstance(self.content, list) and self.role != "user":
            raise ValueError("multimodal content parts are only accepted on user messages")
        if self.role in {"system", "user", "tool"} and self.content is None:
            raise ValueError(f"{self.role} messages require content")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.role == "assistant" and self.content is None and not self.tool_calls:
            raise ValueError("assistant messages require content or tool_calls")
        return self


class OpenAIJSONSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: NonEmptyText = Field(max_length=64)
    description: str | None = Field(default=None, max_length=4096)
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool = False

    @field_validator("schema_")
    @classmethod
    def validate_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            make_spec(name="schema", schema=value, strict=False)
        except StructuredSchemaError as exc:
            raise ValueError(str(exc)) from exc
        return value

    @model_validator(mode="after")
    def validate_strict_schema(self):
        try:
            make_spec(name=self.name, schema=self.schema_, strict=self.strict)
        except StructuredSchemaError as exc:
            raise ValueError(str(exc)) from exc
        return self

class OpenAIResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text", "json_object", "json_schema"]
    json_schema: OpenAIJSONSchema | None = None

class OpenAIChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat request plus explicit Gopi serving extensions."""

    model_config = ConfigDict(extra="ignore")

    model: NonEmptyText
    messages: list[OpenAIChatMessage] = Field(
        min_length=1,
        max_length=256,
    )

    stream: bool = False
    response_format: OpenAIResponseFormat | None = None
    reasoning_effort: Literal["none", "low", "medium", "high"] = "none"

    # Gopi serving extensions. They make the browser UI and OpenAI-compatible
    # clients share one fully-featured contract without changing standard fields.
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]{1,128}$")
    mode: Literal["balanced", "creative", "precise", "coding"] = "balanced"
    web_search: bool = False
    rag: bool = False
    mcp: bool = False
    mcp_server: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]{1,128}$")
    attachments: list[TextAttachment] = Field(default_factory=list, max_length=8)
    no_repeat_ngram_size: int = Field(default=3, ge=0, le=16)
    min_tokens: int = Field(default=1, ge=0, le=1_000_000)

    max_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    max_completion_tokens: int | None = Field(
        default=None,
        ge=1,
        le=1_000_000,
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        allow_inf_nan=False,
    )

    top_p: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    top_k: int = Field(default=40, ge=0, le=100_000)

    min_p: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )

    stop: str | list[str] | None = None

    seed: int | None = Field(
        default=None,
        ge=0,
        le=2**63 - 1,
    )
    repetition_penalty: float = Field(default=1.1, ge=0.1, le=2.0)
    # These OpenAI parameters are intentionally represented so clients receive
    # a deterministic validation error instead of having them silently ignored.
    # The current sampler does not implement presence/frequency penalties.
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    user: str | None = Field(
        default=None,
        max_length=128,
    )

    tools: list[OpenAITool] | None = Field(
        default=None,
        max_length=128,
    )

    tool_choice: OpenAIToolChoice = "auto"

    @model_validator(mode="after")
    def reject_unsupported_penalties(self):
        unsupported = []
        if self.presence_penalty is not None:
            unsupported.append("presence_penalty")
        if self.frequency_penalty is not None:
            unsupported.append("frequency_penalty")
        if unsupported:
            raise ValueError(
                "unsupported generation parameters: " + ", ".join(unsupported)
            )
        return self

    @model_validator(mode="after")
    def validate_tool_choice_contract(self):
        names = {tool.function.name for tool in (self.tools or [])}
        if self.tool_choice == "required" and not names:
            raise ValueError("tool_choice=required requires at least one tool")
        if isinstance(self.tool_choice, OpenAIToolChoiceObject) and self.tool_choice.function.name not in names:
            raise ValueError(f"tool_choice references unknown tool: {self.tool_choice.function.name}")
        return self

    @field_validator("messages")
    @classmethod
    def validate_messages(
        cls,
        messages: list[OpenAIChatMessage],
    ) -> list[OpenAIChatMessage]:
        if not any(message.role == "user" for message in messages):
            # Tool continuation requests can technically contain no user
            # message, so accept assistant/tool conversations when a tool
            # result is present.
            has_tool_result = any(
                message.role == "tool" for message in messages
            )

            if not has_tool_result:
                raise ValueError(
                    "messages must contain at least one user message"
                )

        return messages

    @field_validator("stop")
    @classmethod
    def normalize_stop(
        cls,
        value: str | list[str] | None,
    ) -> str | list[str] | None:
        if value is None:
            return None

        values = [value] if isinstance(value, str) else value
        normalized: list[str] = []

        for item in values:
            if not item:
                raise ValueError("stop sequences cannot be empty")

            if len(item) > 1_024:
                raise ValueError(
                    "stop sequences cannot exceed 1024 characters"
                )

            if item not in normalized:
                normalized.append(item)

        return normalized[0] if isinstance(value, str) else normalized

    @field_validator("tools")
    @classmethod
    def validate_tools(
        cls,
        values: list[OpenAITool] | None,
    ) -> list[OpenAITool] | None:
        if values is None:
            return None

        names: set[str] = set()

        for tool in values:
            name = tool.function.name

            if name in names:
                raise ValueError(f"duplicate tool name: {name}")

            names.add(name)

        return values

    @field_validator("tool_choice")
    @classmethod
    def validate_tool_choice(
        cls,
        value: OpenAIToolChoice,
    ) -> OpenAIToolChoice:
        return value

    def generation_request(
        self,
        expected_model: str,
    ) -> GenerateRequest:
        if self.model != expected_model:
            raise ValueError(f"unknown model: {self.model}")

        tool_names = {tool.function.name for tool in (self.tools or [])}
        if self.tool_choice == "required" and not tool_names:
            raise ValueError("tool_choice=required requires at least one tool")
        if isinstance(self.tool_choice, OpenAIToolChoiceObject):
            selected = self.tool_choice.function.name
            if selected not in tool_names:
                raise ValueError(f"tool_choice references unknown tool: {selected}")

        maximum = (
            self.max_completion_tokens
            or self.max_tokens
            or 128
        )

        stops = (
            [self.stop]
            if isinstance(self.stop, str)
            else list(self.stop or [])
        )

        latest_user = next(
            (
                message.text_content()
                for message in reversed(self.messages)
                if message.role == "user" and message.text_content()
            ),
            None,
        )

        # A continuation containing tool results may not have a user message.
        # In that case use the latest non-empty message as the prompt fallback.
        if latest_user is None:
            latest_user = next(
                (
                    message.text_content()
                    for message in reversed(self.messages)
                    if message.text_content()
                ),
                None,
            )

        if not latest_user:
            raise ValueError(
                "messages must contain usable message content"
            )

        request = GenerateRequest(
            prompt=latest_user,
            session_id=self.session_id,
            mode=self.mode,
            max_tokens=maximum,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            min_tokens=self.min_tokens,
            seed=self.seed,
            stop=stops,
            chat_tools=self.tools or [],
            tool_choice=self.tool_choice,
            response_format=(self.response_format.model_dump(by_alias=True, mode="json") if self.response_format else None),
            reasoning_effort=self.reasoning_effort,
            web_search=self.web_search,
            rag=self.rag,
            mcp=self.mcp,
            mcp_server=self.mcp_server,
            attachments=self.attachments,
        )

        request._chat_messages = [
            {
                "role": message.role,
                "content": (
                    message.content
                    if isinstance(message.content, str)
                    else [part.model_dump(mode="json") for part in (message.content or [])]
                ),
                **(
                    {"tool_calls": [
                        tool.model_dump(mode="json")
                        for tool in message.tool_calls
                    ]}
                    if message.tool_calls
                    else {}
                ),
                **(
                    {"tool_call_id": message.tool_call_id}
                    if message.tool_call_id
                    else {}
                ),
            }
            for message in self.messages
        ]

        return request


class GenerateResponse(StrictSchema):
    id: str
    object: Literal["text_generation"] = "text_generation"
    created: int
    model: str
    bot_name: str
    text: str
    finish_reason: FinishReason
    usage: TokenUsage
    reasoning_content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class OpenAIChatCompletionMessage(StrictSchema):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    refusal: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class OpenAIChatCompletionChoice(StrictSchema):
    index: int
    message: OpenAIChatCompletionMessage
    finish_reason: str


class OpenAIChatCompletionResponse(StrictSchema):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChatCompletionChoice]
    usage: TokenUsage
    incomplete_details: dict[str, Any] | None = None


class OpenAIChatCompletionDelta(StrictSchema):
    role: Literal["assistant"] | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class OpenAIChatCompletionChunkChoice(StrictSchema):
    index: int
    delta: OpenAIChatCompletionDelta
    finish_reason: str | None = None


class OpenAIChatCompletionChunk(StrictSchema):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[OpenAIChatCompletionChunkChoice]


class WorkspaceAction(StrictSchema):
    type: Literal["read", "search", "edit", "patch", "test", "git"]
    path: str = Field(default="", max_length=1024)
    query: str = Field(default="", max_length=4096)
    content: str = Field(default="", max_length=262_144)
    expected_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    apply: bool = False
    preset: Literal["unit", "all"] = "unit"
    operation: Literal[
        "status",
        "diff",
        "diff_staged",
        "log",
    ] = "status"


class WorkspaceAgentRequest(StrictSchema):
    actions: list[WorkspaceAction] = Field(
        min_length=1,
        max_length=8,
    )


class WorkspaceAgentResponse(StrictSchema):
    results: list[dict]


class SessionSummary(StrictSchema):
    session_id: str
    updated: float
    message_count: int = Field(ge=0)
    title: str


class SessionListResponse(StrictSchema):
    sessions: list[SessionSummary]


class SessionMemoryResponse(StrictSchema):
    session_id: str
    messages: list[dict[str, Any]]


class SessionDeleteResponse(StrictSchema):
    session_id: str
    deleted: bool


class TrainingReviewResponse(StrictSchema):
    session_id: str
    prompt: str
    answer: str


class TrainingApprovalRequest(StrictSchema):
    approved: bool = True
    corrected_response: str | None = Field(default=None, max_length=262_144)


class TrainingApprovalResponse(StrictSchema):
    session_id: str
    approved: bool
    example_count: int = Field(ge=0)


class TrainingDeleteResponse(StrictSchema):
    session_id: str
    deleted_count: int = Field(ge=0)


class HealthResponse(StrictSchema):
    status: Literal["ok", "ready", "not_ready"]
    service: str
    version: str
    model: str
    ready: bool
    authentication_required: bool


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


class StreamToolCallEvent(StrictSchema):
    type: Literal["tool_call"] = "tool_call"
    id: str
    tool_call: OpenAIToolCall


class StreamDoneEvent(StrictSchema):
    type: Literal["done"] = "done"
    id: str
    finish_reason: FinishReason
    usage: TokenUsage


class StreamErrorEvent(StrictSchema):
    type: Literal["error"] = "error"
    id: str | None = None
    error: ErrorDetail


class OpenAIModel(StrictSchema):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "gopi"
    capabilities: dict[str, Any] = Field(default_factory=dict)
    architecture: str | None = None
    context_length: int | None = Field(default=None, ge=0)


class OpenAIModelList(StrictSchema):
    object: Literal["list"] = "list"
    data: list[OpenAIModel]


# Pydantic forward-reference resolution.
GenerateRequest.model_rebuild()
