import json

from scripts.prepare_recovery_sft import (
    SYSTEM_PROMPT,
    build_dataset,
    extract_pair,
    quality_pair,
    record_has_quality_signals,
)


def write_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_extract_pair_uses_last_completed_user_assistant_turn() -> None:
    pair = extract_pair({"messages": [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "two"},
    ]})
    assert pair == ("second", "two")


def test_quality_filter_rejects_placeholders_and_heavy_repetition() -> None:
    assert quality_pair(("Who are you?", "I am {name}.")) is None
    repeated = "word one two " * 20
    assert quality_pair(("Explain this", repeated)) is None
    assert quality_pair(("What is 2 + 2?", "4")) == ("What is 2 + 2?", "4")


def test_helpsteer_requires_good_helpfulness_correctness_and_coherence() -> None:
    good = {"scores": {"helpfulness": 3, "correctness": 4, "coherence": 3}}
    bad = {"scores": {"helpfulness": 2, "correctness": 4, "coherence": 4}}
    assert record_has_quality_signals(good, "helpsteer")
    assert not record_has_quality_signals(bad, "helpsteer")
    assert record_has_quality_signals({}, "core_chat")


def test_builder_standardizes_prompt_and_prevents_split_overlap(tmp_path) -> None:
    data_root = tmp_path / "processed"
    record = {"messages": [
        {"role": "system", "content": "Different prompt"},
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": "Hello!"},
    ]}
    write_records(data_root / "core_chat" / "validation.jsonl", [record])
    write_records(data_root / "core_chat" / "train.jsonl", [record, {
        "prompt": "Held out in another domain", "response": "Do not train on this."
    }, {
        "prompt": "What is your name?", "response": "I am Gopi."
    }])
    write_records(data_root / "general_qa" / "validation.jsonl", [{
        "prompt": "Held out in another domain", "response": "Validation answer."
    }])
    output = tmp_path / "recovery"
    summary = build_dataset(data_root, output)

    chat_train = [json.loads(line) for line in (output / "chat/train.jsonl").read_text().splitlines()]
    chat_validation = [json.loads(line) for line in (output / "chat/validation.jsonl").read_text().splitlines()]
    assert summary["domains"]["chat"]["train"] == 1
    assert chat_train[0]["messages"][0]["content"] == SYSTEM_PROMPT
    assert chat_validation[0]["messages"][0]["content"] == SYSTEM_PROMPT
    assert {item["messages"][1]["content"] for item in chat_train}.isdisjoint(
        {item["messages"][1]["content"] for item in chat_validation}
    )
    assert (output / "chat/dataset-manifest.yaml").is_file()
