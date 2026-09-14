import pytest

from scripts.audit_data_quality import analyze_record, prompt_key, record_text
from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer


def tokenizer():
    pieces = [*DEFAULT_SPECIAL_TOKENS, *BYTE_ENCODER.values()]
    vocab = {piece: i for i, piece in enumerate(pieces)}
    return Tokenizer(vocab, special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS})


def test_audit_counts_chat_template_and_detects_truncated_supervision():
    tok = tokenizer()
    record = {"messages": [{"role": "user", "content": "a" * 100},
                           {"role": "assistant", "content": "answer"}]}
    result = analyze_record(record, tok, 32)
    assert result["truncated"]
    assert result["supervised_tokens"] == 0
    assert result["missing_system"]
    assert result["tokens"] > len(tok.encode(record_text(record)))


def test_audit_scores_chosen_not_rejected_preference_answer():
    record = {"prompt": "question", "chosen": "good", "rejected": "bad" * 1000}
    result = analyze_record(record, tokenizer(), 64)
    assert result["chat"] and not result["truncated"]
    assert result["prompts"] == ["question"]
    assert "bad" not in record_text(record)


def test_audit_flags_missing_assistant_and_normalizes_prompt_identity():
    result = analyze_record({"messages": [{"role": "user", "content": "hello"}]}, tokenizer(), 64)
    assert result["invalid_chat"]
    assert prompt_key(" Ｈello\n WORLD ") == prompt_key("hello world")
    with pytest.raises(ValueError, match="object"):
        analyze_record([], tokenizer(), 64)
