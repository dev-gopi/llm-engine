import pytest

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
