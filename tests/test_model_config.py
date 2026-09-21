import pytest
import torch

from model.config import estimate_model_size, normalize_model_config
from model.gpt import MiniGPT


def test_common_large_model_aliases_are_backward_compatible() -> None:
    config = normalize_model_config({
        "vocab_size": 32000,
        "model_dim": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "context_length": 8192,
        "intermediate_size": 11008,
    })
    assert config["hidden_size"] == 4096
    assert config["layers"] == 32
    assert config["heads"] == 32
    assert config["kv_heads"] == 8
    assert config["max_position"] == 8192
    assert config["ffn_hidden_size"] == 11008


def test_conflicting_aliases_are_rejected() -> None:
    with pytest.raises(ValueError, match="conflicting"):
        normalize_model_config({"hidden_size": 256, "model_dim": 512})


def test_model_builds_from_common_alias_names() -> None:
    model = MiniGPT.from_config({
        "vocab_size": 64,
        "model_dim": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "context_length": 128,
        "intermediate_size": 48,
        "position_type": "rotary",
        "ffn_activation": "swiglu",
    })
    assert model.dim == 16
    assert len(model.blocks) == 2
    assert model.max_positions == 128


def test_estimator_handles_7b_shape_without_allocating_weights() -> None:
    size = estimate_model_size({
        "vocab_size": 32000,
        "hidden_size": 4096,
        "layers": 36,
        "heads": 32,
        "kv_heads": 8,
        "max_position": 32768,
        "position_type": "rotary",
        "ffn_hidden_size": 11008,
        "ffn_activation": "swiglu",
        "attention_bias": False,
        "ffn_bias": False,
        "norm_type": "rms_norm",
        "norm_bias": False,
        "tie_word_embeddings": True,
    })
    assert 6_000_000_000 < size.parameters < 8_000_000_000
    assert size.parameter_bytes_bf16 == size.parameters * 2


def test_rope_requires_even_head_dimension() -> None:
    with pytest.raises(ValueError, match="even"):
        estimate_model_size({
            "vocab_size": 100,
            "hidden_size": 15,
            "layers": 1,
            "heads": 3,
            "max_position": 16,
            "position_type": "rotary",
        })


def test_trillion_profile_can_be_planned_without_allocating_weights():
    from utils.config import load_yaml
    from training.planner import plan_training
    model = load_yaml("configs/scaling/model.1t.yaml")
    training = load_yaml("configs/scaling/training.1t.yaml")
    size = estimate_model_size(model)
    assert size.parameters == 999_772_348_416
    plan = plan_training(model, training, training_tokens=1_000_000, gpus=1024)
    assert plan.parameters == size.parameters
    with pytest.raises(ValueError, match="planning-only"):
        MiniGPT.from_config(model)


def test_moe_profile_reports_total_and_active_parameters():
    from utils.config import load_yaml
    config = load_yaml("configs/scaling/model.moe-100b.yaml")
    size = estimate_model_size(config)
    assert 90_000_000_000 < size.parameters < 110_000_000_000
    assert 10_000_000_000 < size.active_parameters_per_token < 20_000_000_000
    with pytest.raises(ValueError, match="planning-only"):
        MiniGPT.from_config(config)


def test_small_moe_model_builds_from_config():
    config = {
        "vocab_size": 32, "hidden_size": 16, "layers": 2, "heads": 4,
        "kv_heads": 2, "max_position": 16, "ffn_hidden_size": 32,
        "ffn_type": "moe", "num_experts": 4, "experts_per_token": 2,
        "router_bias": True, "router_jitter": 0.05,
    }
    model = MiniGPT.from_config(config)
    assert len(model.blocks[0].ffn.experts) == 4
    assert model.blocks[0].ffn.experts_per_token == 2
    assert model.blocks[0].ffn.router.bias is not None
    assert model.blocks[0].ffn.router_jitter == 0.05
    assert model(torch.tensor([[1, 2, 3]])).shape == (1, 3, 32)
    assert model.num_parameters() == estimate_model_size(config).parameters


@pytest.mark.parametrize("override", [
    {"ffn_type": "unknown"},
    {"ffn_type": "moe", "num_experts": 0},
    {"ffn_type": "moe", "num_experts": 2, "experts_per_token": 3},
    {"ffn_type": "moe", "num_experts": 2, "router_jitter": -0.1},
])
def test_invalid_moe_config_fails_before_model_construction(override):
    config = {
        "vocab_size": 32, "hidden_size": 16, "layers": 1, "heads": 4,
        "max_position": 16, **override,
    }
    with pytest.raises(ValueError):
        MiniGPT.from_config(config)


def test_hybrid_attention_estimator_counts_only_dense_kv_layers():
    config = {
        "vocab_size": 64,
        "hidden_size": 32,
        "layers": 8,
        "heads": 4,
        "kv_heads": 2,
        "max_position": 1024,
        "position_type": "rotary",
        "attention_layer_pattern": ["linear", "linear", "linear", "dense"],
    }
    hybrid = estimate_model_size(config)
    dense = estimate_model_size({**config, "attention_layer_pattern": ["dense"]})
    assert hybrid.linear_attention_layers == 6
    assert hybrid.full_attention_layers == 2
    assert hybrid.kv_cache_bytes_bf16_per_sequence == dense.kv_cache_bytes_bf16_per_sequence // 4
    assert hybrid.linear_state_bytes_bf16_per_sequence > 0
    assert hybrid.runtime_state_bytes_bf16_per_sequence < dense.runtime_state_bytes_bf16_per_sequence


def test_modern_hybrid_profiles_are_plannable_without_changing_default_model():
    from utils.config import load_yaml
    default = load_yaml("configs/model.gpu.yaml")
    hybrid = load_yaml("configs/model.hybrid.gpu.yaml")
    planned = load_yaml("configs/scaling/model.hybrid-moe-7b.yaml")
    assert "attention_layer_pattern" not in default
    assert hybrid["attention_layer_pattern"] == ["linear", "linear", "linear", "dense"]
    hybrid_size = estimate_model_size(hybrid)
    assert hybrid_size.linear_attention_layers == 12
    assert hybrid_size.full_attention_layers == 4
    planned_size = estimate_model_size(planned)
    assert 6_000_000_000 < planned_size.parameters < 8_000_000_000
    assert planned_size.active_parameters_per_token < planned_size.parameters / 2
    with pytest.raises(ValueError, match="planning-only"):
        MiniGPT.from_config(planned)
