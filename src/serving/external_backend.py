"""Delegated OpenAI-compatible backend for optimized external model runtimes.

This backend lets the Gopi API/agent surface sit in front of runtimes such as
``llama-server``.  It is deliberately transport-only: GGUF parsing, custom
quantization kernels and device placement remain the responsibility of the
specialized runtime, while Gopi retains its API, auth, monitoring and client
interface.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .runtime import BackendGeneration, BackendStreamEvent, BackendUnavailableError, InvalidGenerationRequestError
from .schemas import FinishReason, GenerateRequest, OpenAIToolCall
from .vision_runtime import has_image_input


def _finish_reason(value: Any) -> FinishReason:
    normalized = str(value or "stop").lower()
    if normalized == "length":
        return FinishReason.LENGTH
    if normalized in {"tool_calls", "function_call"}:
        return FinishReason.TOOL_CALLS
    if normalized in {"cancelled", "canceled"}:
        return FinishReason.CANCELLED
    return FinishReason.STOP


class OpenAICompatibleBackend:
    """Forward generation to an OpenAI-compatible local inference server."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8080",
        model: str = "local-model",
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        context_length: int = 0,
        parameter_count: int | None = None,
        supports_tool_calling: bool = True,
        supports_vision: bool = False,
        supports_reasoning: bool = True,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url.rstrip("/")
        if not self.base_url:
            raise ValueError("base_url must not be empty")
        self.model = str(model)
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.context_length = max(0, int(context_length))
        self.parameter_count = int(parameter_count) if parameter_count else None
        self.supports_tool_calling = bool(supports_tool_calling)
        self.supports_vision = bool(supports_vision)
        self.supports_reasoning = bool(supports_reasoning)
        self._ready = False
        self._client: httpx.AsyncClient | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=httpx.Timeout(self.timeout_seconds),
        )
        try:
            response = await self._client.get("/v1/models")
            response.raise_for_status()
        except (httpx.HTTPError, ValueError):
            self._ready = False
            return
        self._ready = True

    async def shutdown(self) -> None:
        self._ready = False
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _messages(self, request: GenerateRequest) -> list[dict[str, Any]]:
        if request._chat_messages:
            return list(request._chat_messages)
        return [{"role": "user", "content": request.prompt}]

    def _payload(self, request: GenerateRequest, *, stream: bool) -> dict[str, Any]:
        if request.chat_tools and not self.supports_tool_calling:
            raise InvalidGenerationRequestError("external backend is not configured for tool calling")
        if request.reasoning_effort != "none" and not self.supports_reasoning:
            raise InvalidGenerationRequestError("external backend is not configured for reasoning")
        if has_image_input(request._chat_messages) and not self.supports_vision:
            raise InvalidGenerationRequestError("external backend is not configured for vision input")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(request),
            "stream": stream,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "seed": request.seed,
        }
        # llama.cpp accepts these sampler extensions. Other compatible servers
        # generally ignore unknown optional fields only if they support them, so
        # omit disabled values rather than sending every internal knob.
        if request.top_k:
            payload["top_k"] = request.top_k
        if request.min_p:
            payload["min_p"] = request.min_p
        if request.repetition_penalty != 1.0:
            payload["repeat_penalty"] = request.repetition_penalty
        if request.stop:
            payload["stop"] = request.stop
        if request.reasoning_effort != "none":
            payload["reasoning_effort"] = request.reasoning_effort
        if request.chat_tools:
            payload["tools"] = [tool.model_dump(mode="json") for tool in request.chat_tools]
            payload["tool_choice"] = (
                request.tool_choice
                if isinstance(request.tool_choice, str)
                else request.tool_choice.model_dump(mode="json")
            )
        response_format = request.response_format
        if isinstance(response_format, dict):
            payload["response_format"] = response_format
        elif response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _require_client(self) -> httpx.AsyncClient:
        if not self._ready or self._client is None:
            raise BackendUnavailableError("external inference backend is not ready")
        return self._client

    async def generate(self, request: GenerateRequest) -> BackendGeneration:
        client = self._require_client()
        try:
            response = await client.post("/v1/chat/completions", json=self._payload(request, stream=False))
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            message = choice.get("message", {})
            content = message.get("content") or ""
            reasoning_content = message.get("reasoning_content") or message.get("reasoning")
            raw_tool_calls = message.get("tool_calls") or []
            tool_calls = tuple(OpenAIToolCall.model_validate(item) for item in raw_tool_calls)
            usage = body.get("usage", {})
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise BackendUnavailableError(f"external inference request failed: {error}") from error
        if not isinstance(content, str):
            raise InvalidGenerationRequestError("external backend returned non-text message content")
        if reasoning_content is not None and not isinstance(reasoning_content, str):
            raise InvalidGenerationRequestError("external backend returned invalid reasoning content")
        return BackendGeneration(
            text=content,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            finish_reason=_finish_reason(choice.get("finish_reason")),
            cached_tokens=int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0),
            reasoning_tokens=int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0),
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
        )

    async def stream(self, request: GenerateRequest) -> AsyncIterator[BackendStreamEvent]:
        client = self._require_client()
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason: FinishReason | None = None
        tool_fragments: dict[int, dict[str, Any]] = {}
        try:
            async with client.stream(
                "POST", "/v1/chat/completions", json=self._payload(request, stream=True)
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                        choice = event.get("choices", [{}])[0]
                    except (json.JSONDecodeError, IndexError, TypeError):
                        continue
                    usage = event.get("usage") or {}
                    prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens) or prompt_tokens)
                    completion_tokens = int(usage.get("completion_tokens", completion_tokens) or completion_tokens)
                    delta = choice.get("delta") or {}
                    token = delta.get("content") or ""
                    if isinstance(token, str) and token:
                        completion_tokens = max(completion_tokens, completion_tokens + 1)
                        yield BackendStreamEvent(token=token, prompt_tokens=prompt_tokens)
                    reasoning_token = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if isinstance(reasoning_token, str) and reasoning_token:
                        yield BackendStreamEvent(
                            reasoning_token=reasoning_token,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                        )
                    for raw_call in delta.get("tool_calls") or []:
                        if not isinstance(raw_call, dict):
                            continue
                        index = int(raw_call.get("index", 0) or 0)
                        target = tool_fragments.setdefault(index, {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })
                        if isinstance(raw_call.get("id"), str):
                            target["id"] += raw_call["id"]
                        function = raw_call.get("function") or {}
                        if isinstance(function.get("name"), str):
                            target["function"]["name"] += function["name"]
                        if isinstance(function.get("arguments"), str):
                            target["function"]["arguments"] += function["arguments"]
                    if choice.get("finish_reason") is not None:
                        finish_reason = _finish_reason(choice["finish_reason"])
        except httpx.HTTPError as error:
            raise BackendUnavailableError(f"external inference stream failed: {error}") from error
        complete_calls: tuple[OpenAIToolCall, ...] = ()
        if tool_fragments:
            try:
                complete_calls = tuple(
                    OpenAIToolCall.model_validate(tool_fragments[index])
                    for index in sorted(tool_fragments)
                )
            except ValueError as error:
                raise BackendUnavailableError(
                    f"external backend returned malformed streamed tool calls: {error}"
                ) from error
        yield BackendStreamEvent(
            finish_reason=finish_reason or FinishReason.STOP,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tool_calls=complete_calls,
        )
