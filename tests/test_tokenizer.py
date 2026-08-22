import json

import pytest

from tokenizer.bpe import BYTE_ENCODER, merge_pair
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer
from tokenizer.trainer import BPETokenizerTrainer


CORPUS = [
    "Hello world! Hello tokenizer.\n",
    "A byte-level tokenizer preserves  spaces, tabs\tand newlines.\n",
    "Unicode works: नमस्ते, తెలుగు, café, 你好, مرحبا, 🌍🚀.",
] * 20


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return BPETokenizerTrainer(vocab_size=300, min_frequency=2).train(CORPUS)


def test_merge_pair_is_non_overlapping():
    assert merge_pair(("a", "a", "a"), ("a", "a")) == ("aa", "a")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain ASCII",
        "  leading and trailing  \n",
        "नमस्ते दुनिया — 你好世界 — مرحبا 🌍",
        "tabs\tand\r\nnewlines",
        "<|eos|> remains ordinary unless explicitly allowed",
    ],
)
def test_round_trip_preserves_text(tokenizer: Tokenizer, text: str):
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_all_bytes_are_representable(tokenizer: Tokenizer):
    for value in range(256):
        assert BYTE_ENCODER[value] in tokenizer.vocab


def test_special_tokens_are_explicit(tokenizer: Tokenizer):
    text = "hello<|eos|>world"
    ordinary_ids = tokenizer.encode(text)
    special_ids = tokenizer.encode(text, allowed_special={"<|eos|>"})
    assert tokenizer.special_tokens["<|eos|>"] not in ordinary_ids
    assert tokenizer.special_tokens["<|eos|>"] in special_ids
    assert tokenizer.decode(ordinary_ids) == text
    assert tokenizer.decode(special_ids) == text
    assert tokenizer.decode(special_ids, skip_special_tokens=True) == "helloworld"


def test_bos_and_eos(tokenizer: Tokenizer):
    identifiers = tokenizer.encode("hello", add_bos=True, add_eos=True)
    assert identifiers[0] == tokenizer.special_tokens["<|bos|>"]
    assert identifiers[-1] == tokenizer.special_tokens["<|eos|>"]


def test_save_and_load_is_stable(tokenizer: Tokenizer, tmp_path):
    artifact = tokenizer.save(tmp_path)
    restored = Tokenizer.load(artifact)
    sample = "Exact persistence ✅\n"
    assert restored.vocab == tokenizer.vocab
    assert restored.bpe.merges == tokenizer.bpe.merges
    assert restored.encode(sample) == tokenizer.encode(sample)
    assert restored.decode(restored.encode(sample)) == sample
    assert json.loads((tmp_path / "tokenizer.json").read_text())["version"] == 1
    assert (tmp_path / "vocab.json").is_file()
    assert (tmp_path / "merges.txt").is_file()


def test_training_is_deterministic():
    first = BPETokenizerTrainer(vocab_size=280, min_frequency=2).train(CORPUS)
    second = BPETokenizerTrainer(vocab_size=280, min_frequency=2).train(CORPUS)
    assert first.vocab == second.vocab
    assert first.bpe.merges == second.bpe.merges


def test_invalid_configuration_and_ids(tokenizer: Tokenizer):
    with pytest.raises(ValueError, match="at least"):
        BPETokenizerTrainer(vocab_size=len(DEFAULT_SPECIAL_TOKENS) + 255)
    with pytest.raises(ValueError, match="outside"):
        tokenizer.decode([tokenizer.vocab_size])
