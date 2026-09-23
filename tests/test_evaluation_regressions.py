import json
import sys
from pathlib import Path

import pytest

from evaluation.benchmarks import (
    BenchmarkCase,
    NeedleInHaystackCase,
    compare_reports,
    score_answer,
)


def test_needle_in_haystack_probe_is_deterministic_and_scores_only_the_passkey():
    probe = NeedleInHaystackCase(4096, "zephyr42", needle_position=0.75)
    prompt = probe.prompt()
    assert prompt == probe.prompt()
    assert probe.needle_token_index == 3072
    assert prompt.index("The passkey is zephyr42.") == 3072 * len("filler ")
    case = probe.benchmark_case()
    assert score_answer("zephyr42", case) == 1
    assert score_answer("zephyr41", case) == 0
    assert score_answer("The passkey is zephyr42", case) == 0


def test_needle_in_haystack_supports_the_2k_probe_size():
    probe = NeedleInHaystackCase(2048, "harbor7", needle_position=0.5)
    assert probe.needle_token_index == 1024
    assert score_answer("harbor7", probe.benchmark_case()) == 1


def test_needle_in_haystack_probe_validates_its_contract():
    with pytest.raises(ValueError, match="at least"):
        NeedleInHaystackCase(127, "key")
    with pytest.raises(ValueError, match="nonempty"):
        NeedleInHaystackCase(2048, "")
    with pytest.raises(ValueError, match="between"):
        NeedleInHaystackCase(2048, "key", needle_position=1.1)


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


def test_gsm8k_case_requires_a_complete_thinking_trace_before_final_answer():
    case = BenchmarkCase(
        "gsm8k", "Solve this.", ("4",), match="final_number", require_thinking_trace=True,
    )
    assert score_answer("<thinking>2 + 2 = 4.</thinking>\n#### 4", case) == 1
    assert score_answer("#### 4", case) == 0
    assert score_answer("<thinking>2 + 2 = 4.</thinking>", case) == 0


def test_reasoning_code_manifest_preserves_all_scoring_controls():
    path = Path("configs/evaluation.reasoning_code.jsonl")
    cases = [BenchmarkCase.from_mapping(json.loads(line)) for line in path.read_text().splitlines()]
    assert [case.category for case in cases] == ["reasoning_math", "reasoning_logic", "code"]
    assert cases[0].require_thinking_trace
    assert cases[0].max_answer_tokens == 96
    assert score_answer("<thinking>7 + 5 = 12.</thinking>\n#### 12", cases[0]) == 1
    assert score_answer("a + b", cases[2]) == 0
    assert score_answer("a % b", cases[2]) == 1


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


def test_instruction_following_manifest_is_versioned_and_deterministically_scored():
    path = Path("configs/evaluation.instruction_following.jsonl")
    cases = [BenchmarkCase.from_mapping(json.loads(line)) for line in path.read_text().splitlines()]

    assert [case.category for case in cases] == [
        "clarification", "refusal", "format_following", "conversation_consistency", "tool_turn",
    ]
    answers = [
        "Which form and field are confusing?",
        "Sorry, I can't help with harming someone.",
        "READY",
        "Gopi",
        "4",
    ]
    assert [score_answer(answer, case) for answer, case in zip(answers, cases, strict=True)] == [1.0] * 5


def test_ctx002_long_context_fixtures_cover_all_profiles_and_positions():
    import json
    from pathlib import Path

    profiles = (("2k", 2048), ("4k", 4096), ("8k", 8192))
    paths = [Path(f"data/processed/long_context/{name}/validation.jsonl") for name, _ in profiles]
    if not all(path.is_file() for path in paths):
        pytest.skip("long-context validation fixtures are generated data and are not included in source checkouts")

    for (name, length), path in zip(profiles, paths, strict=True):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [row["context_tokens"] for row in rows] == [length] * 3
        assert [row["needle_position"] for row in rows] == [0.10, 0.50, 0.90]
        assert len({row["passkey"] for row in rows}) == 3


def test_ctx002_base_pretraining_profile_publishes_paired_profiles():
    import yaml
    config = yaml.safe_load(Path("configs/pretraining.gpu.yaml").read_text(encoding="utf-8"))
    profiles = config["long_context_profiles"]
    assert set(profiles) == {"2k", "4k", "8k"}
    assert [profiles[name]["max_position"] for name in ("2k", "4k", "8k")] == [2048, 4096, 8192]


def test_benchmark_cli_rejects_contexts_larger_than_the_model_limit(tmp_path, monkeypatch, capsys):
    """The CLI must validate its model config before constructing probe cases."""
    from scripts.evaluate_benchmarks import main

    model_config = tmp_path / "model.yaml"
    model_config.write_text("max_position: 512\n", encoding="utf-8")
    inference_config = tmp_path / "inference.yaml"
    inference_config.write_text("{}\n", encoding="utf-8")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_benchmarks.py",
            "--checkpoint", str(checkpoint),
            "--model-config", str(model_config),
            "--inference-config", str(inference_config),
            "--tokenizer", str(tokenizer),
            "--long-context-lengths", "1024",
        ],
    )

    with pytest.raises(SystemExit) as exited:
        main()

    assert exited.value.code == 2
    assert "exceeds model max_position 512" in capsys.readouterr().err
