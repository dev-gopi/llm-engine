import json
from types import SimpleNamespace

import pytest
import torch

from datasets.collator import Collator
from inference.chat_session import ChatSession, build_chat_sft_example, format_chat_messages
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


def test_session_memory_retrieval_is_scoped_ranked_and_opt_in(tmp_path):
    b = backend()
    alice = ChatSession(b, tmp_path / "chat.sqlite", "alice")
    bob = ChatSession(b, tmp_path / "chat.sqlite", "bob")
    alice.chat("My favourite colour is green")
    bob.chat("My favourite colour is blue")

    matches = alice.retrieve_memory("what colour do I like?")
    assert matches[0]["content"] == "My favourite colour is green"
    assert "blue" not in str(matches)
    assert alice.retrieve_memory("unrelated") == []
    with pytest.raises(ValueError, match="nonempty"):
        alice.retrieve_memory(" ")


def test_canonical_chat_template_masks_every_non_assistant_token() -> None:
    b = backend()
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Say hello."},
        {"role": "assistant", "content": "Hello!"},
    ]

    assert format_chat_messages(messages) == (
        "<|system|>\nBe concise.\n<|end|>\n"
        "<|user|>\nSay hello.\n<|end|>\n"
        "<|assistant|>\nHello!\n<|end|>\n"
    )
    example = build_chat_sft_example(b.tokenizer, messages)
    batch = Collator(pad_token_id=0)([example])

    assert batch["loss_mask"].sum().item() > 0
    assert torch.equal(batch["labels"].eq(-100), ~batch["loss_mask"])
    supervised = b.tokenizer.decode(batch["input_ids"][0][batch["loss_mask"][0]].tolist())
    assert "Hello!" in supervised
    assert "Be concise." not in supervised
    assert "Say hello." not in supervised
