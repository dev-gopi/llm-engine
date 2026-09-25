"""Unit and regression tests for LM Evaluation Harness support and adapters."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from evaluation.harness import (
    ARCHarnessTask,
    EvaluationHarness,
    GSM8KTask,
    HarnessDoc,
    HarnessModelAdapter,
    HarnessReport,
    HellaSwagTask,
    LambadaTask,
    MiniGPTHarnessAdapter,
    MMLUTask,
    TaskRegistry,
    TruthfulQATask,
    WinograndeTask,
    get_standard_fixtures,
    get_task,
    list_tasks,
    register_task,
)
from model.config import normalize_model_config
from model.gpt import MiniGPT
from tokenizer.encoder import Tokenizer


@pytest.fixture
def sample_tokenizer() -> Tokenizer:
    vocab = {
        "<|pad|>": 0,
        "<|unk|>": 1,
        "<|bos|>": 2,
        "<|eos|>": 3,
        "<|system|>": 4,
        "<|user|>": 5,
        "<|assistant|>": 6,
        "a": 7,
        "b": 8,
        "c": 9,
        "d": 10,
        " ": 11,
        "Paris": 12,
        "France": 13,
        "capital": 14,
        "The": 15,
        "of": 16,
        "is": 17,
        "72": 18,
        "####": 19,
        "42": 20,
    }
    special = {
        "<|pad|>": 0,
        "<|unk|>": 1,
        "<|bos|>": 2,
        "<|eos|>": 3,
        "<|system|>": 4,
        "<|user|>": 5,
        "<|assistant|>": 6,
    }
    return Tokenizer(vocab, merges=(), special_tokens=special, tokenizer_type="character")


@pytest.fixture
def small_model(sample_tokenizer: Tokenizer) -> MiniGPT:
    config = normalize_model_config({
        "vocab_size": sample_tokenizer.vocab_size,
        "hidden_size": 32,
        "layers": 2,
        "heads": 2,
        "kv_heads": 1,
        "max_position": 64,
        "ffn_hidden_size": 64,
    })
    return MiniGPT.from_config(config, device="cpu")


def test_harness_adapter_loglikelihood_computes_probabilities_and_greedy_flag(
    small_model: MiniGPT, sample_tokenizer: Tokenizer
):
    adapter = HarnessModelAdapter(small_model, sample_tokenizer, device="cpu", batch_size=2)
    requests = [
        ("The capital of France is", " Paris"),
        ("The capital of France is", " London"),
        ("", "Paris"),
    ]
    results = adapter.loglikelihood(requests)
    assert len(results) == 3
    for logprob, is_greedy in results:
        assert isinstance(logprob, float)
        assert logprob <= 0.0 or torch.isclose(torch.tensor(logprob), torch.tensor(0.0), atol=1e-3)
        assert isinstance(is_greedy, bool)


def test_harness_adapter_loglikelihood_rolling_computes_sequence_logprobs(
    small_model: MiniGPT, sample_tokenizer: Tokenizer
):
    adapter = HarnessModelAdapter(small_model, sample_tokenizer, device="cpu", max_length=16)
    long_text = "The capital of France is Paris " * 5
    results = adapter.loglikelihood_rolling([long_text, "Short text"])
    assert len(results) == 2
    assert isinstance(results[0], float)
    assert isinstance(results[1], float)
    assert results[0] <= 0.0


def test_harness_adapter_generate_until_respects_stop_sequences(
    small_model: MiniGPT, sample_tokenizer: Tokenizer
):
    adapter = HarnessModelAdapter(small_model, sample_tokenizer, device="cpu")
    requests = [
        ("The capital of France is", {"until": ["Paris", "\n"], "max_gen_toks": 10}),
        ("The", {"max_gen_toks": 5}),
    ]
    completions = adapter.generate_until(requests)
    assert len(completions) == 2
    assert isinstance(completions[0], str)
    assert isinstance(completions[1], str)
    assert "Paris" not in completions[0]


def test_mmlu_task_multiple_choice_scoring(small_model: MiniGPT, sample_tokenizer: Tokenizer):
    docs = [
        HarnessDoc(
            query="What is the capital of France?",
            choices=("London", "Paris", "Berlin"),
            target="Paris",
            subject="geography",
        )
    ]
    task = MMLUTask(docs)
    adapter = HarnessModelAdapter(small_model, sample_tokenizer, device="cpu")
    result = task.evaluate(adapter)

    assert result["samples"] == 1
    assert "accuracy" in result
    assert "acc_norm" in result
    assert result["accuracy"] in (0.0, 1.0)
    assert len(result["details"]) == 1
    assert result["details"][0]["target_index"] == 1


def test_gsm8k_task_math_extraction_and_scoring():
    task = GSM8KTask()
    doc = HarnessDoc(query="What is 20 + 22?", target="20 + 22 = 42\n#### 42")

    # Match with #### format
    assert task._score_generation("Let's calculate: 20+22=42\n#### 42", doc.target, doc)
    # Match with final number in text
    assert task._score_generation("The final answer is 42.", doc.target, doc)
    # Mismatch
    assert not task._score_generation("The answer is 100.", doc.target, doc)


def test_arc_and_hellaswag_and_winogrande_and_truthfulqa_task_formatting():
    arc = ARCHarnessTask(get_standard_fixtures("arc_challenge"))
    text = arc.doc_to_text(arc.docs[0])
    assert "Question:" in text
    assert "Answer:" in text

    hs = HellaSwagTask(get_standard_fixtures("hellaswag"))
    assert hs.doc_to_text(hs.docs[0]) == hs.docs[0].query

    wino = WinograndeTask(get_standard_fixtures("winogrande"))
    assert "Sentence:" in wino.doc_to_text(wino.docs[0])

    tqa = TruthfulQATask(get_standard_fixtures("truthfulqa"))
    assert "Q:" in tqa.doc_to_text(tqa.docs[0])


def test_task_registry_operations():
    registry = TaskRegistry()
    tasks = registry.list_tasks()
    assert "mmlu" in tasks
    assert "gsm8k" in tasks
    assert "arc_challenge" in tasks
    assert "lambada" in tasks

    task_obj = registry.get("mmlu")
    assert isinstance(task_obj, MMLUTask)

    class CustomTask(MMLUTask):
        name = "custom_test_task"

    registry.register("custom_test_task", CustomTask)
    assert "custom_test_task" in registry.list_tasks()

    with pytest.raises(KeyError, match="not found"):
        registry.get("nonexistent_task_xyz")


def test_evaluation_harness_runner_produces_structured_report(
    small_model: MiniGPT, sample_tokenizer: Tokenizer, tmp_path: Path
):
    adapter = HarnessModelAdapter(small_model, sample_tokenizer, device="cpu")
    harness = EvaluationHarness()

    report = harness.run(
        adapter,
        tasks=["mmlu", "arc_challenge", "lambada"],
        limit=1,
        checkpoint_name="test_checkpoint.pt",
    )

    assert isinstance(report, HarnessReport)
    assert report.checkpoint == "test_checkpoint.pt"
    assert "mmlu_accuracy" in report.summary
    assert "arc_challenge_accuracy" in report.summary
    assert "lambada_perplexity" in report.summary
    assert report.summary["tasks_evaluated"] == 3

    # Test JSON saving and round-trip
    out_file = tmp_path / "harness_report.json"
    report.save_json(out_file)
    assert out_file.exists()

    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["checkpoint"] == "test_checkpoint.pt"
    assert loaded["summary"]["tasks_evaluated"] == 3


def test_fewshot_context_formatting():
    task = MMLUTask()
    docs = [
        HarnessDoc(query="Q1", choices=("A", "B"), target="A"),
        HarnessDoc(query="Q2", choices=("C", "D"), target="D"),
        HarnessDoc(query="Q3", choices=("E", "F"), target="E"),
    ]
    prompt = task.fewshot_context(docs[2], num_fewshot=2, fewshot_docs=docs)
    assert "Q1" in prompt
    assert "Q2" in prompt
    assert "Q3" in prompt


def test_harness_adapter_truncation_handling(small_model: MiniGPT, sample_tokenizer: Tokenizer):
    # max_length is 16
    adapter = HarnessModelAdapter(small_model, sample_tokenizer, device="cpu", max_length=16, truncation=True)
    long_prefix = "The capital of France is Paris " * 10
    results = adapter.loglikelihood([(long_prefix, " Paris")])
    assert len(results) == 1
    assert isinstance(results[0][0], float)

    adapter_no_trunc = HarnessModelAdapter(small_model, sample_tokenizer, device="cpu", max_length=16, truncation=False)
    results_no_trunc = adapter_no_trunc.loglikelihood([(long_prefix, " Paris")])
    assert len(results_no_trunc) == 1
    assert isinstance(results_no_trunc[0][0], float)


def test_harness_adapter_instance_unpacking(small_model: MiniGPT, sample_tokenizer: Tokenizer):
    class MockInstance:
        def __init__(self, args):
            self.args = args

    adapter = HarnessModelAdapter(small_model, sample_tokenizer, device="cpu")
    inst = MockInstance(("The capital of France is", " Paris"))
    results = adapter.loglikelihood([inst])
    assert len(results) == 1

    rolling_inst = MockInstance(("The capital of France is Paris",))
    rolling_results = adapter.loglikelihood_rolling([rolling_inst])
    assert len(rolling_results) == 1

    gen_inst = MockInstance(("The capital", {"max_gen_toks": 3}))
    gen_results = adapter.generate_until([gen_inst])
    assert len(gen_results) == 1


def test_evaluate_harness_cli_list_tasks(capsys):
    from scripts.evaluate_harness import main
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["evaluate_harness.py", "--list-tasks"]
        main()
        captured = capsys.readouterr()
        assert "Available harness benchmark tasks:" in captured.out
        assert "mmlu" in captured.out
        assert "gsm8k" in captured.out
    finally:
        sys.argv = old_argv

