import json

import pytest
import torch

from inference.pretrained import MiniGPTBackend
from model.gpt import MiniGPT
from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer
from training.checkpoint import save_checkpoint


def make_backend():
    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: index for index, piece in enumerate(pieces)}
    tokenizer = Tokenizer(vocab, special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS})
    config = dict(vocab_size=len(vocab), hidden_size=16, layers=1, heads=2,
                  max_position=64, tie_word_embeddings=True)
    return MiniGPTBackend(MiniGPT.from_config(config), tokenizer, config, device="cpu",
                          generation_config={"max_tokens": 3, "temperature": 0})


def test_bundle_roundtrip_preserves_weights_tying_and_generation(tmp_path):
    backend = make_backend()
    expected = backend.generate("hello")
    path = backend.save_pretrained(tmp_path / "bundle")
    loaded = MiniGPTBackend.from_pretrained(path, device="cpu")
    assert loaded.generate("hello") == expected
    assert loaded.model.head.weight is loaded.model.tok.weight
    for key, value in backend.model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value)
    assert loaded.generate_batch(["hello"])[0] == expected
    assert "".join(event.token for event in loaded.stream("hello")) == expected.text
    with pytest.raises(FileExistsError):
        backend.save_pretrained(path)


def test_bundle_rejects_fingerprint_mismatch(tmp_path):
    path = make_backend().save_pretrained(tmp_path / "bundle")
    config = json.loads((path / "config.json").read_text())
    config["tokenizer_fingerprint"] = "wrong"
    (path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="fingerprint"):
        MiniGPTBackend.from_pretrained(path)


def test_backend_loads_existing_training_checkpoint(tmp_path):
    backend = make_backend()
    path = save_checkpoint(tmp_path / "latest.pt", backend.model,
                           metadata={"tokenizer_fingerprint": backend.tokenizer.fingerprint})
    restored = MiniGPTBackend.from_checkpoint(path, model_config=backend.model_config,
                                              tokenizer=backend.tokenizer, device="cpu",
                                              generation_config=backend.generation_config)
    assert restored.generate("hello") == backend.generate("hello")
