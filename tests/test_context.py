from inference.context import ConversationMemory, SQLiteSessionStore
from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer


def tokenizer() -> Tokenizer:
    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: index for index, piece in enumerate(pieces)}
    return Tokenizer(vocab, special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS})


def test_context_memory_renders_roles_and_preserves_system_prompt() -> None:
    memory = ConversationMemory(tokenizer(), max_tokens=100, system_prompt="You are Gopi.")
    memory.add("user", "Hello")
    memory.add("assistant", "Hi!")
    prompt = memory.render(add_generation_prompt=True, reserve_tokens=10)
    assert prompt.startswith("<|system|>")
    assert "<|user|>\nHello" in prompt
    assert prompt.endswith("<|assistant|>\n")
    memory.clear()
    assert [message.role for message in memory.snapshot()] == ["system"]


def test_context_memory_trims_oldest_non_system_messages() -> None:
    memory = ConversationMemory(tokenizer(), max_tokens=45, system_prompt="Gopi")
    memory.add("user", "first message")
    memory.add("assistant", "first answer")
    memory.add("user", "new")
    roles_and_text = [(message.role, message.content) for message in memory.snapshot()]
    assert ("system", "Gopi") in roles_and_text
    assert roles_and_text[-1] == ("user", "new")


def test_sqlite_session_store_persists_messages(tmp_path) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite", tokenizer(), max_tokens=100, system_prompt="Gopi")
    memory = store.load("abc")
    memory.add("user", "Remember this")
    store.save("abc", memory)
    restored = store.load("abc")
    assert restored.snapshot()[-1].content == "Remember this"
    store.delete("abc")
    assert len(store.load("abc").snapshot()) == 1


def test_context_memory_handles_large_reserve_tokens() -> None:
    memory = ConversationMemory(tokenizer(), max_tokens=100, system_prompt="You are Gopi.")
    memory.add("user", "Hello world")
    prompt = memory.render(add_generation_prompt=True, reserve_tokens=100)
    assert "<|system|>" in prompt
    assert "Hello world" in prompt


def test_context_memory_rejects_negative_reserve_tokens() -> None:
    import pytest
    memory = ConversationMemory(tokenizer(), max_tokens=100, system_prompt="You are Gopi.")
    memory.add("user", "Hello")
    with pytest.raises(ValueError, match="reserve_tokens must be non-negative"):
        memory.render(reserve_tokens=-1)

