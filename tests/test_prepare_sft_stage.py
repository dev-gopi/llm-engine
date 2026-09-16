import json

import pytest
import yaml

from scripts.prepare_sft_stage import prepare, record_key, refresh_training_config, rejection_reason
from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer


def tokenizer():
    pieces = [*DEFAULT_SPECIAL_TOKENS, *BYTE_ENCODER.values()]
    vocab = {piece: i for i, piece in enumerate(pieces)}
    return Tokenizer(vocab, special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS})


def chat(prompt, answer="OK"):
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}


def test_preparation_preserves_sft_masks_and_reserves_all_heldout_prompts(tmp_path):
    tok = tokenizer()
    tok.save(tmp_path / "tokenizer")
    source = tmp_path / "source"
    source.mkdir()
    train = source / "train.jsonl"
    validation = source / "validation.jsonl"
    train_rows = [chat("held out"), chat("too long held out"), chat("unique", "def f():\n    return 1"),
                  chat("unique", "duplicate answer"), chat("test only"), chat("eval only")]
    train.write_text("".join(json.dumps(row) + "\n" for row in train_rows))
    original = train.read_bytes()
    validation.write_text(json.dumps(chat("held out")) + "\n" + json.dumps(chat("too long held out", "x" * 200)) + "\n")
    (source / "test.jsonl").write_text(json.dumps(chat("test only")) + "\n")
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"prompt": "eval only"}) + "\n")
    config = {"train_files": [str(train)], "validation_files": [str(validation)],
              "validation_domains": {"chat": [str(validation)]}, "dataset_weights": {"source": 1},
              "max_sequence_length": 128, "runtime": {"tokenizer": str(tmp_path / "tokenizer")}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "cleaned"
    summary = prepare(config_path, output, cases=[cases])
    kept = [json.loads(line) for line in (output / "source/train.jsonl").read_text().splitlines()]
    assert kept == [train_rows[2]]
    assert train.read_bytes() == original
    assert summary["files"][1]["heldout_overlap"] == 4
    generated = yaml.safe_load((output / "training.yaml").read_text())
    assert generated["prepared_data"]["tokenizer_fingerprint"] == tok.fingerprint
    assert generated["require_init_from"] and generated["require_prepared_data"]
    assert generated["validation_domains"]["chat"] == generated["validation_files"]
    with pytest.raises(ValueError, match="already exists"):
        prepare(config_path, output)


def test_preparation_rejects_unsupported_tools_and_keeps_code_whitespace():
    tok = tokenizer()
    malformed = {"messages": [{"role": "tool", "content": "result"}]}
    assert rejection_reason(malformed, tok, 128) == "invalid_chat"
    assert rejection_reason(chat("question", "x" * 200), tok, 128) == "overlength_chat"
    assert rejection_reason(chat("code", "def f():\n    return 1"), tok, 128) is None
    assert record_key(chat("ＡＢＣ")) == record_key(chat("abc"))


def test_refresh_training_config_reuses_audited_files(tmp_path):
    tok = tokenizer()
    tok.save(tmp_path / "tokenizer")
    source = tmp_path / "source"
    source.mkdir()
    train = source / "train.jsonl"
    validation = source / "validation.jsonl"
    train.write_text(json.dumps(chat("train")) + "\n")
    validation.write_text(json.dumps(chat("validation")) + "\n")
    config = {
        "train_files": [str(train)], "validation_files": [str(validation)],
        "validation_domains": {"chat": [str(validation)]},
        "dataset_weights": {"source": 1}, "max_sequence_length": 128,
        "batch_size": 4, "runtime": {"tokenizer": str(tmp_path / "tokenizer")},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "cleaned"
    prepare(config_path, output)
    config["batch_size"] = 1
    config_path.write_text(yaml.safe_dump(config))

    destination = refresh_training_config(config_path, output)
    refreshed = yaml.safe_load(destination.read_text())

    assert refreshed["batch_size"] == 1
    assert refreshed["train_files"] == [str((output / "source/train.jsonl").resolve())]
    assert refreshed["prepared_data"]["tokenizer_fingerprint"] == tok.fingerprint
