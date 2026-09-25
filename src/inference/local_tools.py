"""Small deterministic tools that are safe to expose to generation requests."""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_TOOL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")


class ToolCallError(ValueError):
    """Raised when generated tool-call text is not valid under its schema."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


def parse_tool_call(text: str, schema: Mapping[str, Any] | None = None) -> ToolCall:
    """Parse exactly one `<tool_call>` envelope and validate its JSON payload."""
    match = re.fullmatch(r"\s*<tool_call>(.*?)</tool_call>\s*", text, flags=re.DOTALL)
    if match is None:
        raise ToolCallError("tool call must use one <tool_call> JSON envelope")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as error:
        raise ToolCallError("tool call contains invalid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
        raise ToolCallError("tool call requires exactly name and arguments")
    name, arguments = payload["name"], payload["arguments"]
    if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
        raise ToolCallError("tool name is invalid")
    if not isinstance(arguments, dict):
        raise ToolCallError("tool arguments must be an object")
    if schema is not None:
        validate_json_schema(arguments, schema)
    return ToolCall(name, arguments)


def validate_json_schema(value: Any, schema: Mapping[str, Any]) -> None:
    """Validate the safe JSON-schema subset used by local and MCP tools."""
    expected = schema.get("type")
    valid_types = {
        "object": dict, "array": list, "string": str, "integer": int,
        "number": (int, float), "boolean": bool, "null": type(None),
    }
    if expected in valid_types:
        accepted = valid_types[expected]
        if not isinstance(value, accepted) or (expected in {"integer", "number"} and isinstance(value, bool)):
            raise ToolCallError(f"expected {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolCallError("value is not an allowed enum member")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise ToolCallError("object schema properties must be an object")
        required = schema.get("required", [])
        if not isinstance(required, list) or any(not isinstance(key, str) for key in required):
            raise ToolCallError("object schema required must be a string list")
        missing = [key for key in required if key not in value]
        if missing:
            raise ToolCallError("missing required argument: " + ", ".join(missing))
        if schema.get("additionalProperties") is False:
            unknown = set(value) - set(properties)
            if unknown:
                raise ToolCallError("unexpected argument: " + sorted(unknown)[0])
        for key, item in value.items():
            if key in properties:
                validate_json_schema(item, properties[key])
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for item in value:
            validate_json_schema(item, schema["items"])


def calculate(expression: str) -> int | float:
    """Evaluate a bounded arithmetic expression without executing Python code."""
    if not expression.strip() or len(expression) > 256:
        raise ValueError("calculation must contain 1 to 256 characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("invalid calculation") from error

    def evaluate(node: ast.AST, depth: int = 0):
        if depth > 20:
            raise ValueError("calculation is too complex")
        if isinstance(node, ast.Expression):
            return evaluate(node.body, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            if not math.isfinite(float(node.value)) or abs(node.value) > 1e100:
                raise ValueError("number is outside the supported range")
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left, right = evaluate(node.left, depth + 1), evaluate(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
                raise ValueError("exponent is outside the supported range")
            try:
                result = _BINARY_OPERATORS[type(node.op)](left, right)
            except (ArithmeticError, OverflowError) as error:
                raise ValueError("calculation could not be completed") from error
            if not math.isfinite(float(result)) or abs(result) > 1e100:
                raise ValueError("result is outside the supported range")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand, depth + 1))
        raise ValueError("only arithmetic operators and numbers are supported")

    return evaluate(tree)


def tool_context(prompt: str, tools: list[str], *, now: datetime | None = None) -> str:
    """Return trusted tool results for inclusion in a user turn."""
    results: list[str] = []
    if "calculator" in tools:
        lowered = prompt.lower()
        prefix_length = 10 if lowered.startswith("/calculate") else 5 if lowered.startswith("/calc") else 0
        expression = prompt[prefix_length:].strip() if prefix_length else _find_expression(prompt)
        if expression:
            results.append(f"Calculator result: {calculate(expression)}")
    if "datetime" in tools:
        current = (now or datetime.now().astimezone())
        results.append(f"Current local date and time: {current.isoformat(timespec='seconds')} ({current.tzname()})")
    if not results:
        return prompt
    return f"{prompt}\n\nTrusted tool results (use these to answer):\n" + "\n".join(results)


def direct_tool_answer(prompt: str, tools: list[str], *, now: datetime | None = None) -> str | None:
    """Return exact answers for deterministic tool requests without model generation."""
    normalized = prompt.strip()
    lowered = normalized.lower()
    calculator_shortcut = lowered.startswith(("/calculate ", "/calc "))
    if calculator_shortcut or "calculator" in tools:
        prefix_length = 10 if lowered.startswith("/calculate") else 5 if lowered.startswith("/calc") else 0
        expression = normalized[prefix_length:].strip() if prefix_length else _find_expression(normalized)
        # Markdown copy/paste may escape arithmetic symbols and leave sentence punctuation.
        expression = expression.replace("\\*", "*").replace("\\+", "+").replace("\\-", "-").replace("\\/", "/").rstrip(".")
        if expression:
            try:
                return f"{expression} = {calculate(expression)}"
            except ValueError as error:
                return f"Calculator error: {error}. Example: /calc (25 + 5) * 4"
        if calculator_shortcut:
            return "Calculator error: enter an arithmetic expression. Example: /calc (25 + 5) * 4"
    if lowered in {"/time", "/date", "/datetime"}:
        current = now or datetime.now().astimezone()
        return f"Current local date and time: {current.strftime('%A, %d %B %Y at %I:%M:%S %p')} ({current.tzname()})"
    return None


def _find_expression(prompt: str) -> str:
    """Extract a likely arithmetic expression from conversational text."""
    candidates = re.findall(r"[\d.][\d.eE\s()+\-*/%]*", prompt)
    expressions = [candidate.strip().rstrip(".") for candidate in candidates]
    expressions = [value for value in expressions if re.search(r"[+\-*/%]", value)]
    return max(expressions, key=len, default="")

@dataclass(frozen=True)
class ToolApprovalPolicy:
    allowed_tools: frozenset[str]
    require_human_approval: bool = True
    max_steps: int = 8

    def check(self, name: str) -> None:
        if name not in self.allowed_tools:
            raise PermissionError(f"tool {name!r} is not approved for agent execution")
