from datetime import datetime, timezone

import pytest

from inference.local_tools import (
    ToolCallError,
    calculate,
    direct_tool_answer,
    parse_tool_call,
    tool_context,
)


def test_calculator_evaluates_arithmetic() -> None:
    assert calculate("(12 + 3) * 2") == 30
    assert calculate("2 ** 8") == 256


@pytest.mark.parametrize("expression", ["__import__('os')", "2 ** 1000", "[1, 2]", "1 / 0"])
def test_calculator_rejects_unsafe_or_unbounded_input(expression: str) -> None:
    with pytest.raises(ValueError):
        calculate(expression)


def test_tool_context_includes_selected_results() -> None:
    result = tool_context(
        "6 / 2", ["calculator", "datetime"],
        now=datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc),
    )
    assert "Calculator result: 3.0" in result
    assert "2026-08-27T12:00:00+00:00" in result


def test_calculator_understands_chat_and_ignores_unrelated_prompts() -> None:
    assert "Calculator result: 42" in tool_context("What is 6 * 7?", ["calculator"])
    assert tool_context("Tell me a story", ["calculator"]) == "Tell me a story"


def test_direct_calculator_answer_handles_markdown_escapes() -> None:
    assert direct_tool_answer(r"/calc 25 \* 4.", []) == "25 * 4 = 100"


@pytest.mark.parametrize("prompt", ["/calc hello", "/calc 1 / 0", "/calc 2 +"])
def test_direct_calculator_returns_friendly_errors(prompt: str) -> None:
    answer = direct_tool_answer(prompt, [])
    assert answer is not None
    assert answer.startswith("Calculator error:")


def test_direct_datetime_answer_is_human_readable() -> None:
    answer = direct_tool_answer(
        "/time", [], now=datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)
    )
    assert answer == "Current local date and time: Thursday, 27 August 2026 at 12:00:00 PM (UTC)"


def test_tool_call_parser_requires_one_valid_envelope_and_schema() -> None:
    schema = {
        "type": "object", "properties": {"expression": {"type": "string"}},
        "required": ["expression"], "additionalProperties": False,
    }
    call = parse_tool_call(
        '<tool_call>{"name":"calculator","arguments":{"expression":"6 * 7"}}</tool_call>', schema,
    )
    assert call.name == "calculator"
    assert call.arguments == {"expression": "6 * 7"}
    for value in (
        "not JSON",
        '<tool_call>{"name":"calculator","arguments":[]}</tool_call>',
        '<tool_call>{"name":"calculator","arguments":{"extra":1}}</tool_call>',
    ):
        with pytest.raises(ToolCallError):
            parse_tool_call(value, schema)


def test_tool_call_parser_has_no_syntax_failures_for_repeated_valid_calls() -> None:
    for value in range(100):
        call = parse_tool_call(
            f'<tool_call>{{"name":"calculator","arguments":{{"expression":"{value} + 1"}}}}</tool_call>'
        )
        assert call.arguments["expression"] == f"{value} + 1"
