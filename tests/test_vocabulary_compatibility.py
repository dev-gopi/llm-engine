import pytest

from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer


def make_tokenizer() -> Tokenizer:
    vocab = {token: index for index, token in enumerate(DEFAULT_SPECIAL_TOKENS)}
    for value in range(256):
        vocab[BYTE_ENCODER[value]] = len(vocab)
    return Tokenizer(
        vocab,
        special_tokens={token: vocab[token] for token in DEFAULT_SPECIAL_TOKENS},
    )


def test_exact_tokenizer_keeps_configured_vocabulary():
    tokenizer = make_tokenizer()
    config = {"vocab_size": tokenizer.vocab_size, "hidden_size": 8}
    assert adapt_config_to_tokenizer(config, tokenizer) == config
    assert checkpoint_tokenizer_options(tokenizer)["allow_vocab_extension"] is False


def test_append_only_tokenizer_resizes_config_and_accepts_base_checkpoint():
    base = make_tokenizer()
    extended = base.extend([" বাংলা", "👨‍👩‍👧‍👦"])
    config = adapt_config_to_tokenizer({"vocab_size": base.vocab_size}, extended)
    options = checkpoint_tokenizer_options(extended)

    assert config["vocab_size"] == extended.vocab_size
    assert base.fingerprint in options["compatible_tokenizer_fingerprints"]
    assert options["allow_vocab_extension"] is True


def test_unrelated_vocabulary_size_is_rejected():
    tokenizer = make_tokenizer()
    with pytest.raises(ValueError, match="not a verified append-only extension"):
        adapt_config_to_tokenizer({"vocab_size": tokenizer.vocab_size - 1}, tokenizer)
