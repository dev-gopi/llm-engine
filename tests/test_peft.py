import torch

from model.gpt import MiniGPT
from training.checkpoint import load_checkpoint, save_checkpoint
from training.peft import LoRALinear, apply_lora


def _model() -> MiniGPT:
    return MiniGPT(vocab_size=32, dim=16, layers=2, heads=4, kv_heads=2,
                   max_pos=32, position_type="rotary", ffn_hidden_dim=32)


def test_lora_starts_as_exact_noop_and_only_adapters_are_trainable() -> None:
    torch.manual_seed(7)
    model = _model().eval()
    tokens = torch.randint(0, 32, (2, 8))
    expected = model(tokens)
    details = apply_lora(model, {"method": "lora", "rank": 4, "alpha": 8,
                                 "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]})

    torch.testing.assert_close(model(tokens), expected, rtol=0, atol=0)
    assert details["matched_modules"]
    assert all("lora_" in name for name, parameter in model.named_parameters()
               if parameter.requires_grad)
    assert 0 < details["trainable_parameters"] < details["total_parameters"]


def test_lora_checkpoint_reconstructs_adapters_automatically(tmp_path) -> None:
    model = _model()
    config = {"method": "lora", "rank": 2, "alpha": 4, "dropout": 0.0,
              "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"]}
    metadata = apply_lora(model, config)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_b.fill_(0.01)
    path = save_checkpoint(tmp_path / "lora.pt", model, metadata={"peft": metadata})
    restored = _model()

    state = load_checkpoint(path, restored, restore_rng=False)

    assert state["metadata"]["peft"]["method"] == "lora"
    assert any(isinstance(module, LoRALinear) for module in restored.modules())
    for expected, actual in zip(model.state_dict().values(), restored.state_dict().values()):
        torch.testing.assert_close(actual, expected)
