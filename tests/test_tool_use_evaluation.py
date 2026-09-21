import json
from pathlib import Path

import pytest

from inference.local_tools import ToolCallError, calculate, validate_json_schema


def test_tool_use_fixture_covers_required_reliability_cases() -> None:
    cases = [json.loads(line) for line in Path("configs/evaluation.tool_use.jsonl").read_text().splitlines()]
    assert [case["id"] for case in cases] == [
        "calculator_selection", "schema_rejection", "result_handling",
        "error_recovery", "parallel_read_intent", "permission_denial",
    ]
    assert cases[4]["mode"] == "parallel_intent"

    for case in cases:
        for call in case["calls"]:
            if call["name"] not in case["allowed_tools"]:
                assert case["expected_error"] == "not allowlisted"
                continue
            if case["id"] == "schema_rejection":
                with pytest.raises(ToolCallError, match="expected string"):
                    validate_json_schema(call["arguments"], call["schema"])
                continue
            validate_json_schema(call["arguments"], call["schema"])

    assert calculate(cases[0]["calls"][0]["arguments"]["expression"]) == 42
    assert calculate(cases[2]["calls"][0]["arguments"]["expression"]) == 4
    with pytest.raises(ValueError, match="could not be completed"):
        calculate(cases[3]["calls"][0]["arguments"]["expression"])
