import json
from types import SimpleNamespace

import pytest

from inference.chat_session import ChatSession
from inference.generator import GenerationResult
from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer


def backend():
    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: i for i, piece in enumerate(pieces)}
    tokenizer = Tokenizer(vocab, special_tokens={p: vocab[p] for p in DEFAULT_SPECIAL_TOKENS})
    return SimpleNamespace(tokenizer=tokenizer, generator=SimpleNamespace(max_positions=256),
                           generation_config={"max_tokens": 16},
                           generate=lambda *a, **k: GenerationResult("Hello!", (1,), 1, "stop"))


def test_history_survives_restart_and_requires_explicit_export_approval(tmp_path):
    b = backend()
    session = ChatSession(b, tmp_path / "chat.sqlite", "alice")
    session.chat("My name is Alice")
    restarted = ChatSession(b, tmp_path / "chat.sqlite", "alice")
    assert restarted.history() == session.history()
    with pytest.raises(ValueError, match="no approved"):
        session.export_training(tmp_path / "train.jsonl")
    session.approve_conversation(corrected_response="Hello Alice!")
    assert restarted.export_training(tmp_path / "train.jsonl") == 1
    row = json.loads((tmp_path / "train.jsonl").read_text())
    assert row["messages"][-1]["content"] == "Hello Alice!"
    assert session.history()[-1]["content"] == "Hello!"
    with pytest.raises(FileExistsError):
        restarted.export_training(tmp_path / "train.jsonl")
    other = ChatSession(b, tmp_path / "chat.sqlite", "bob")
    assert not other.history()
    with pytest.raises(ValueError, match="no approved"):
        other.export_training(tmp_path / "other.jsonl")
    restarted.forget(include_training_examples=True)
    assert not restarted.history()
    with pytest.raises(ValueError, match="no approved"):
        restarted.export_training(tmp_path / "deleted.jsonl")


def test_failed_generation_does_not_persist_partial_turn(tmp_path):
    b = backend()
    session = ChatSession(b, tmp_path / "chat.sqlite", "a")
    session.chat("hi")
    before = session.history()

    def fail(*a, **k):
        raise RuntimeError("generation failed")

    b.generate = fail
    with pytest.raises(RuntimeError):
        session.chat("new message")
    assert session.history() == before
    with pytest.raises(ValueError, match="no unreviewed"):
        session.approve_conversation()


def test_history_is_used_and_training_export_is_loadable(tmp_path):
    from datasets.loader import TextDataset
    b = backend()
    session = ChatSession(b, tmp_path / "chat.sqlite", "a")
    session.chat("My name is Alice")
    prompts = []

    def respond(prompt, **options):
        prompts.append(prompt)
        assert len(b.tokenizer.encode(prompt, add_bos=True, allowed_special="all")) + 16 < 256
        return GenerationResult("Alice", (1,), 1, "stop")

    b.generate = respond
    session.chat("What is my name?")
    assert "My name is Alice" in prompts[0]
    session.approve_conversation()
    path = tmp_path / "train.jsonl"
    session.export_training(path)
    dataset = TextDataset.from_files([path], b.tokenizer, max_length=256)
    assert len(dataset) == 1
    assert dataset[0]["loss_mask"].any()
