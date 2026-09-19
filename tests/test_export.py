import torch
from safetensors import safe_open

from model.gpt import MiniGPT
from scripts.export import export_model


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
