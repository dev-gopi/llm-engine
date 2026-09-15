import pytest

from evaluation.benchmarks import BenchmarkCase, compare_reports, score_answer


def test_expression_scoring_preserves_operators():
    case = BenchmarkCase("coding", "Remainder expression?", ("a % b", "a%b"), match="exact_code")
    assert score_answer("a + b", case) == 0
    assert score_answer("a b", case) == 0
    assert score_answer("```python\na % b\n```", case) == 1


def test_numeric_scoring_does_not_accept_incidental_correct_number():
    case = BenchmarkCase("math", "Reply with the number only.", ("13",), match="number")
    assert score_answer("13 then 9. Final answer: 9", case) == 0
    assert score_answer("-13", case) == 0
    assert score_answer("13.", case) == 1


def test_final_number_requires_explicit_final_answer():
    case = BenchmarkCase("gsm8k", "Solve this.", ("1234",), match="final_number")
    assert score_answer("1234 is an intermediate result. #### 45", case) == 0
    assert score_answer("Reasoning.\n#### 1,234", case) == 1
    assert score_answer("1234", case) == 0


def test_contains_match_rejects_overlong_or_identity_leaking_answer():
    case = BenchmarkCase(
        "chat", "Greet me", ("hello",), ("open assistant",),
        max_answer_tokens=4,
    )
    assert score_answer("Hello, nice to meet you", case) == 0
    assert score_answer("Hello from Open Assistant", case) == 0
    assert score_answer("Hello there", case) == 1


def report(scores, protocol=None):
    return {"protocol": protocol or {"cases": "fixed"}, "results": [
        {"category": category, "prompt": str(i), "score": score}
        for i, (category, score) in enumerate(scores)
    ]}


def test_retention_gate_rejects_improvement_that_forgets_a_passing_case():
    before = report([("english", 1), ("math", 0), ("math", 0)])
    after = report([("english", 0), ("math", 1), ("math", 1)])
    comparison = compare_reports(before, after)
    assert not comparison["passed"]
    assert comparison["accuracy_delta"] > 0
    assert comparison["domain_accuracy_deltas"]["english"] == -1


def test_retention_gate_accepts_identical_answers_and_rejects_changed_protocol():
    value = report([("english", 1), ("math", 0)])
    assert compare_reports(value, value)["passed"]
    with pytest.raises(ValueError, match="protocol"):
        compare_reports(value, {**value, "protocol": {"cases": "changed"}})
    with pytest.raises(ValueError, match="coverage"):
        compare_reports(value, report([("english", 1)]))
