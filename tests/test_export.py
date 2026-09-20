import hashlib
import json

import torch
from safetensors import safe_open

from model.gpt import MiniGPT
from scripts.export import export_model, write_export_manifest
from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer


def test_safetensors_and_pytorch_export(tmp_path) -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8).eval()
    safe_path = export_model(model, tmp_path / "model.safetensors", "safetensors", sequence_length=4)
    with safe_open(safe_path, framework="pt") as artifact:
        assert "head.weight" in artifact.keys()
    program_path = export_model(model, tmp_path / "model.pt2", "torch_export", sequence_length=4)
    exported = torch.export.load(program_path).module()
    assert exported(torch.tensor([[1, 2, 3, 4]])).shape == (1, 4, 16)


def test_int4_safetensors_export_is_self_describing(tmp_path) -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8).eval()
    path = export_model(model, tmp_path / "model-int4.safetensors", "safetensors", weight_dtype="int4")
    with safe_open(path, framework="pt") as artifact:
        assert "head.weight.int4_packed" in artifact.keys()
        assert artifact.metadata()["format"] == "llm-engine.int4.v1"


def test_export_manifest_is_content_addressed_and_includes_tokenizer_compatibility(tmp_path) -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8).eval()
    artifact = export_model(model, tmp_path / "model.safetensors", "safetensors")
    config = {"vocab_size": 16, "hidden_size": 8}
    (tmp_path / "model.yaml").write_text("vocab_size: 16\nhidden_size: 8\n", encoding="utf-8")
    vocab = {piece: index for index, piece in enumerate(list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values()))}
    tokenizer = Tokenizer(vocab, special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS})
    tokenizer.save(tmp_path / "tokenizer")

    path = write_export_manifest(artifact, config=config, tokenizer=tokenizer, export_format="safetensors", weight_dtype="float32")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["artifacts"]["model.safetensors"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert payload["tokenizer"]["fingerprint"] == tokenizer.fingerprint
    assert payload["model_config"]["values"] == config
