"""Native chat, tool-calling, and explicit reasoning wire protocols.

The project serves both its own decoder and delegated OpenAI-compatible
backends.  This module keeps the native textual protocol deterministic while
leaving delegated requests in their original OpenAI shape.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from jsonschema import exceptions as jsonschema_exceptions
from jsonschema.validators import validator_for

from .schemas import (
    OpenAITool,
    OpenAIToolCall,
    OpenAIToolChoiceObject,
    OpenAIToolFunctionCall,
)

_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"
_TOOL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_THINKING_PATTERN = re.compile(r"<thinking>\s*(.*?)\s*</thinking>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ParsedToolCalls:
    content: str
    tool_calls: tuple[OpenAIToolCall, ...]
    error: str | None = None


def flatten_content(content: Any, *, image_placeholder: str = "[image]") -> str:
    """Flatten an OpenAI message content value for the native text model."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    pieces: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = str(part.get("type", ""))
        if kind == "text" and isinstance(part.get("text"), str):
            pieces.append(part["text"])
        elif kind == "image_url":
            pieces.append(image_placeholder)
    return "\n".join(piece for piece in pieces if piece).strip()


def _serialized_tool_call(call: dict[str, Any]) -> str:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    arguments: Any = function.get("arguments", "{}")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = arguments
    payload = {
        "id": call.get("id"),
        "name": function.get("name"),
        "arguments": arguments,
    }
    return f"{_TOOL_OPEN}{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}{_TOOL_CLOSE}"


def normalize_native_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert OpenAI chat history into text turns understood by the native model."""
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = flatten_content(message.get("content"))
        calls = message.get("tool_calls")
        if role == "assistant" and isinstance(calls, list) and calls:
            call_text = "\n".join(
                _serialized_tool_call(call) for call in calls if isinstance(call, dict)
            )
            content = "\n".join(part for part in (content, call_text) if part)
        if role == "tool":
            call_id = str(message.get("tool_call_id") or "unknown")
            payload = content or "(empty tool result)"
            content = f"<tool_result id={json.dumps(call_id)}>{payload}</tool_result>"
        if not content:
            # OpenAI permits assistant messages whose only payload is tool_calls;
            # after normalization that case is non-empty. Other empty turns add
            # no useful model context and are skipped.
            continue
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        # Prevent user content from forging role-token boundaries.
        content = content.replace("<|", "< |")
        normalized.append({"role": role, "content": content})
    return normalized


def render_native_messages(
    system_prompt: str,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Render native chat history, including tool result turns."""
    turns = [{"role": "system", "content": system_prompt}, *normalize_native_messages(messages)]
    chunks: list[str] = []
    for turn in turns:
        role = turn["role"]
        content = turn["content"].strip()
        if not content:
            continue
        chunks.append(f"<|{role}|>\n{content}\n")
    if add_generation_prompt:
        chunks.append("<|assistant|>\n")
    return "".join(chunks)


def build_tool_system_instruction(tools: list[OpenAITool], tool_choice: Any) -> str:
    """Build a compact, deterministic native function-calling contract."""
    if not tools or tool_choice == "none":
        return ""
    definitions = [
        {
            "name": tool.function.name,
            "description": tool.function.description or "",
            "parameters": tool.function.parameters or {"type": "object", "properties": {}},
        }
        for tool in tools
    ]
    selected = selected_tool_for_forced_choice(tools, tool_choice)
    if selected is not None:
        choice = (
            f"You MUST call the function {selected.function.name!r}. "
            "Return ONLY its JSON arguments object. Do not include a function name, tags, Markdown, or prose."
        )
    elif tool_choice == "required":
        choice = (
            "You MUST call one available function. Return ONLY a JSON object with exactly the top-level "
            'shape {"name":"function_name","arguments":{...}}. Do not add tags, Markdown, or prose.'
        )
    else:
        choice = "Call a function only when it is useful; otherwise answer normally."
    contract = ""
    if selected is None and tool_choice != "required":
        contract = (
            " When calling a function, emit only one or more blocks exactly in this form: "
            '<tool_call>{"name":"function_name","arguments":{}}</tool_call>. '
        )
    return (
        "Function calling is available. " + choice + contract
        + " arguments must be a JSON object satisfying that function's JSON Schema. "
        + "Do not invent function names. Available functions: "
        + json.dumps(definitions, ensure_ascii=False, separators=(",", ":"))
    )


def selected_tool_for_forced_choice(tools: list[OpenAITool], tool_choice: Any) -> OpenAITool | None:
    """Return the single function whose invocation is already determined."""
    if isinstance(tool_choice, OpenAIToolChoiceObject):
        name = tool_choice.function.name
        return next((tool for tool in tools if tool.function.name == name), None)
    if tool_choice == "required" and len(tools) == 1:
        return tools[0]
    return None


def forced_tool_json_schema(tools: list[OpenAITool], tool_choice: Any) -> dict[str, Any] | None:
    """Return a decoding schema for tool_choice modes that require a call."""
    selected = selected_tool_for_forced_choice(tools, tool_choice)
    if selected is not None:
        schema = selected.function.parameters or {"type": "object", "properties": {}}
        # Function parameters are required by the OpenAI contract to describe
        # an arguments object. Reject malformed schemas at request parsing time
        # where possible; keep a safe object fallback here for compatibility.
        return schema if isinstance(schema, dict) else {"type": "object"}
    if tool_choice == "required" and tools:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "arguments": {"type": "object"},
            },
            "required": ["name", "arguments"],
        }
    return None


def build_reasoning_system_instruction(reasoning_effort: str) -> str:
    if reasoning_effort == "none":
        return ""
    return (
        "For this request, an explicit reasoning trace is enabled. If you reason step by step, "
        "put that trace inside exactly one <thinking>...</thinking> block, then put the final "
        "answer after </thinking>. Never put the final answer inside the thinking block."
    )


def split_reasoning_trace(text: str) -> tuple[str, str | None]:
    """Separate a well-formed explicit reasoning trace from the final answer."""
    if not isinstance(text, str) or "<thinking>" not in text.lower():
        return text.strip(), None
    matches = list(_THINKING_PATTERN.finditer(text))
    if len(matches) != 1:
        return text.strip(), None
    match = matches[0]
    # Only accept a trace if no unmatched/nested tags remain outside the match.
    before = text[: match.start()].strip()
    after = text[match.end() :].strip()
    reasoning = match.group(1).strip()
    remaining = f"{before}\n{after}".strip()
    lower_remaining = remaining.lower()
    if not reasoning or "<thinking>" in lower_remaining or "</thinking>" in lower_remaining:
        return text.strip(), None
    return remaining, reasoning


def _validate_arguments(tool: OpenAITool, arguments: dict[str, Any]) -> str | None:
    schema = tool.function.parameters or {"type": "object"}
    try:
        validator_class = validator_for(schema)
        validator_class.check_schema(schema)
        validator = validator_class(schema)
        validator.validate(arguments)
    except jsonschema_exceptions.SchemaError as exc:
        return f"tool schema for {tool.function.name} is invalid: {exc.message}"
    except jsonschema_exceptions.ValidationError as exc:
        return f"arguments for {tool.function.name} do not satisfy its schema: {exc.message}"
    return None


def _candidate_payloads(text: str, *, allow_bare_json: bool) -> tuple[list[Any], str]:
    matches = list(_TOOL_PATTERN.finditer(text))
    payloads: list[Any] = []
    if matches:
        for match in matches:
            try:
                payloads.append(json.loads(match.group(1)))
            except ValueError as exc:
                payloads.append({"__parse_error__": str(exc)})
        visible = _TOOL_PATTERN.sub("", text).strip()
        return payloads, visible

    stripped = text.strip()
    if allow_bare_json and (stripped.startswith("{") or stripped.startswith("[")):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return list(parsed), ""
            return [parsed], ""
        except ValueError:
            pass
    return [], text.strip()


def parse_tool_calls(text: str, tools: list[OpenAITool], tool_choice: Any) -> ParsedToolCalls:
    """Parse and schema-validate native model function-call output."""
    if not tools or tool_choice == "none":
        return ParsedToolCalls(text.strip(), ())
    by_name = {tool.function.name: tool for tool in tools}
    forced_choice = tool_choice == "required" or isinstance(tool_choice, OpenAIToolChoiceObject)
    payloads, visible = _candidate_payloads(text, allow_bare_json=forced_choice)
    if not payloads:
        return ParsedToolCalls(visible, ())

    selected_name: str | None = None
    if isinstance(tool_choice, OpenAIToolChoiceObject):
        selected_name = tool_choice.function.name
    elif tool_choice == "required" and len(tools) == 1:
        selected_name = tools[0].function.name

    calls: list[OpenAIToolCall] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            return ParsedToolCalls(visible, tuple(calls), "tool call payload must be a JSON object")
        if "__parse_error__" in payload:
            return ParsedToolCalls(visible, tuple(calls), f"tool call JSON is invalid: {payload['__parse_error__']}")

        function = payload.get("function") if isinstance(payload.get("function"), dict) else None
        name = payload.get("name") or (function or {}).get("name") or selected_name
        arguments: Any = payload.get("arguments", (function or {}).get("arguments"))

        # For a forced/single required function, allow the model to emit the
        # arguments object directly. This works well with JSON-constrained
        # decoding while the server supplies the already-selected function name.
        if name is None and selected_name is not None:
            name = selected_name
            arguments = payload
        elif name == selected_name and arguments is None and not any(
            key in payload for key in ("name", "function", "arguments")
        ):
            arguments = payload

        if not isinstance(name, str) or name not in by_name:
            return ParsedToolCalls(visible, tuple(calls), f"unknown tool in model output: {name!r}")
        if selected_name is not None and name != selected_name:
            return ParsedToolCalls(visible, tuple(calls), f"model called {name!r} but {selected_name!r} was required")
        if arguments is None:
            arguments = {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError as exc:
                return ParsedToolCalls(visible, tuple(calls), f"arguments for {name} are not valid JSON: {exc}")
        if not isinstance(arguments, dict):
            return ParsedToolCalls(visible, tuple(calls), f"arguments for {name} must be a JSON object")
        error = _validate_arguments(by_name[name], arguments)
        if error is not None:
            return ParsedToolCalls(visible, tuple(calls), error)
        call_id = payload.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"call_{uuid.uuid4().hex}"
        calls.append(OpenAIToolCall(
            id=call_id,
            function=OpenAIToolFunctionCall(
                name=name,
                arguments=json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
            ),
        ))
    return ParsedToolCalls(visible, tuple(calls))
