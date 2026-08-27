from datetime import datetime, timezone

import pytest

from inference.local_tools import calculate, direct_tool_answer, tool_context


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
