"""Small deterministic tools that are safe to expose to generation requests."""

from __future__ import annotations

import ast
import math
import operator
import re
from datetime import datetime


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
