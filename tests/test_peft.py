import torch

from inference.generator import Generator
from model.gpt import MiniGPT
from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer
from training.checkpoint import load_checkpoint, save_checkpoint
from training.peft import LoRALinear, apply_lora, load_lora_adapter, lora_adapter_state_dict, merge_and_unload


def _model() -> MiniGPT:
    return MiniGPT(vocab_size=32, dim=16, layers=2, heads=4, kv_heads=2,
                   max_pos=32, position_type="rotary", ffn_hidden_dim=32)


def _tokenizer() -> Tokenizer:
    pieces = [*DEFAULT_SPECIAL_TOKENS, *BYTE_ENCODER.values()]
    vocab = {piece: index for index, piece in enumerate(pieces)}
    return Tokenizer(vocab, special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS})


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


def test_merged_lora_model_matches_active_adapter_and_has_no_peft_modules() -> None:
    torch.manual_seed(8)
    model = _model().eval()
    apply_lora(model, {"rank": 2, "alpha": 4, "target_modules": ["q_proj", "v_proj"]})
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_b.normal_(std=0.01)
    tokens = torch.randint(0, 32, (2, 8))
    expected = model(tokens)

    merged = merge_and_unload(model).eval()

    torch.testing.assert_close(merged(tokens), expected)
    assert not any(isinstance(module, LoRALinear) for module in merged.modules())


def test_lora_adapter_state_can_be_swapped_without_changing_base_weights() -> None:
    model = _model()
    apply_lora(model, {"rank": 2, "alpha": 4, "target_modules": ["q_proj"]})
    first = lora_adapter_state_dict(model)
    second = {name: value + 1 for name, value in first.items()}
    base = next(module.base.weight.detach().clone() for module in model.modules() if isinstance(module, LoRALinear))

    load_lora_adapter(model, second)

    for name, value in second.items():
        torch.testing.assert_close(lora_adapter_state_dict(model)[name], value)
    actual_base = next(module.base.weight for module in model.modules() if isinstance(module, LoRALinear))
    torch.testing.assert_close(actual_base, base)


def test_generator_swaps_lora_adapter_and_invalidates_prefix_cache() -> None:
    model = _model()
    apply_lora(model, {"rank": 2, "alpha": 4, "target_modules": ["q_proj"]})
    generator = Generator(model, _tokenizer(), device="cpu", prefix_cache_capacity=1)
    adapter = {name: value + 0.25 for name, value in lora_adapter_state_dict(model).items()}
    generator.prefix_cache.put((1,), "adapter-dependent logits")

    generator.swap_lora_adapter(adapter)

    assert generator.prefix_cache is None
    for name, value in adapter.items():
        torch.testing.assert_close(lora_adapter_state_dict(model)[name], value)
    generator.swap_lora_adapter(None)
    for name, value in generator._base_lora_adapter.items():
        torch.testing.assert_close(lora_adapter_state_dict(model)[name], value)
