import pytest
import torch

from model.gpt import MiniGPT
from model.positional import PositionalEmbedding, SinusoidalPositionalEmbedding


def test_default_positions_and_shape() -> None:
    module = PositionalEmbedding(8, 4)
    tokens = torch.zeros((2, 3), dtype=torch.long)
    expected = module.weight[:3].unsqueeze(0).expand(2, -1, -1)
    assert torch.equal(module(tokens), expected)


def test_explicit_positions_broadcast_and_offset() -> None:
    module = PositionalEmbedding(8, 4)
    tokens = torch.zeros((2, 2), dtype=torch.long)
    expected = module.weight[3:5].unsqueeze(0).expand(2, -1, -1)
    assert torch.equal(module(tokens, position_offset=3), expected)
    ids = torch.tensor([4, 1])
    expected = module.weight[ids].unsqueeze(0).expand(2, -1, -1)
    assert torch.equal(module(tokens, position_ids=ids), expected)


def test_padding_uses_compact_positions_and_returns_zero_vectors() -> None:
    module = PositionalEmbedding(8, 3)
    tokens = torch.zeros((2, 4), dtype=torch.long)
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 0]])
    output = module(tokens, attention_mask=mask)
    assert torch.count_nonzero(output[0, :2]) == 0
    assert torch.count_nonzero(output[1, 3]) == 0
    assert torch.equal(output[0, 2:], module.weight[:2])
    assert torch.equal(output[1, :3], module.weight[:3])


def test_bounds_and_input_validation() -> None:
    module = PositionalEmbedding(4, 2)
    tokens = torch.zeros((1, 2), dtype=torch.long)
    with pytest.raises(IndexError, match="position IDs"):
        module(tokens, position_offset=3)
    with pytest.raises(ValueError, match="non-negative"):
        module(tokens, position_offset=-1)
    with pytest.raises(TypeError, match="integer dtype"):
        module(tokens, position_ids=torch.tensor([0.0, 1.0]))


def test_resize_preserves_existing_weights_and_initializes_new_rows() -> None:
    module = PositionalEmbedding(3, 4, initializer_range=0.01)
    original = module.weight.detach().clone()
    module.resize(6)
    assert module.max_positions == 6
    assert torch.equal(module.weight[:3], original)
    assert module.weight.shape == (6, 4)
    assert torch.count_nonzero(module.weight[3:]) > 0


def test_from_config_and_gradient_flow() -> None:
    module = PositionalEmbedding.from_config(
        {"max_position": 5, "hidden_size": 3, "initializer_range": 0.01}
    )
    module(torch.zeros((2, 4), dtype=torch.long)).sum().backward()
    assert module.weight.grad is not None
    assert module.weight.shape == (5, 3)


def test_gpt_accepts_explicit_position_controls() -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8).eval()
    tokens = torch.tensor([[1, 2]])
    with torch.no_grad():
        offset_logits = model(tokens, position_offset=2)
        explicit_logits = model(tokens, position_ids=torch.tensor([2, 3]))
    assert torch.allclose(offset_logits, explicit_logits)


def test_sinusoidal_positions_support_odd_embedding_width() -> None:
    module = SinusoidalPositionalEmbedding(8, 7)
    output = module(torch.zeros((2, 3), dtype=torch.long))
    assert output.shape == (2, 3, 7)


def test_rotary_gpt_sizes_cache_for_explicit_position_ids() -> None:
    model = MiniGPT(
        vocab_size=16, dim=8, layers=1, heads=2, max_pos=8, position_type="rotary"
    ).eval()
    tokens = torch.tensor([[1, 2]])
    logits = model(tokens, position_ids=torch.tensor([10, 11]))
    assert logits.shape == (1, 2, 16)


def test_sinusoidal_token_ids_preserve_fractional_embeddings():
    module = SinusoidalPositionalEmbedding(16, 8)
    tokens = torch.tensor([[1, 2, 3]])
    actual = module(tokens, position_offset=2)
    assert actual.is_floating_point()
    torch.testing.assert_close(actual, module.weight[2:5].unsqueeze(0))
    assert (actual != actual.round()).any()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rotary_cache_growth_after_dtype_conversion_matches_fp32(dtype):
    from model.positional import RotaryPositionalEmbedding
    reference = RotaryPositionalEmbedding(16, max_position_embeddings=1024)
    module = RotaryPositionalEmbedding(16, max_position_embeddings=8).to(dtype=dtype)
    x = torch.zeros(1, 2, 1024, 16, dtype=dtype)
    actual = module(x, seq_len=1024)
    expected = reference(x, seq_len=1024)
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target, atol=0, rtol=0)
    module.float()
    for result, target in zip(module(x.float(), seq_len=1024), reference(x.float(), seq_len=1024)):
        torch.testing.assert_close(result, target, atol=0, rtol=0)


@pytest.mark.parametrize("scaling_type", ["ntk", "yarn"])
def test_rope_long_context_scaling_builds_extended_position_tables(scaling_type):
    from model.positional import RotaryPositionalEmbedding
    module = RotaryPositionalEmbedding(
        16, max_position_embeddings=1024, scaling_type=scaling_type,
        scaling_factor=4.0, original_max_position_embeddings=1024,
    )
    cos, sin = module(torch.zeros(1, 1, 4096, 16), seq_len=4096)
    assert cos.shape == sin.shape == (1, 1, 4096, 16)
    assert torch.isfinite(cos).all() and torch.isfinite(sin).all()


def test_yarn_preserves_high_frequency_and_stretches_low_frequency_features():
    from model.positional import RotaryPositionalEmbedding
    baseline = RotaryPositionalEmbedding(16, max_position_embeddings=1024)
    yarn = RotaryPositionalEmbedding(
        16, max_position_embeddings=1024, scaling_type="yarn",
        scaling_factor=4.0, original_max_position_embeddings=1024,
    )
    assert yarn.inv_freq[0] == baseline.inv_freq[0]
    assert yarn.inv_freq[-1] < baseline.inv_freq[-1]


def test_model_config_wires_rope_scaling_into_long_context_model():
    from model.gpt import MiniGPT

    model = MiniGPT.from_config({
        "vocab_size": 32,
        "hidden_size": 16,
        "layers": 1,
        "heads": 2,
        "kv_heads": 1,
        "max_position": 2048,
        "position_type": "rotary",
        "rope_base": 10000.0,
        "rope_scale": 2.0,
        "rope_scaling_type": "ntk",
        "rope_original_max_position": 1024,
        "ffn_hidden_size": 32,
        "ffn_activation": "swiglu",
        "ffn_multiple_of": 8,
        "norm_type": "rms_norm",
        "norm_bias": False,
        "ffn_bias": False,
        "attention_bias": False,
    })
    assert model.max_positions == 2048
    assert model.rotary_emb is not None
    assert model.rotary_emb.scaling_type == "ntk"
    assert model.rotary_emb.scaling_factor == 2.0
    assert model.rotary_emb.original_max_position_embeddings == 1024


def test_ctx002_profile_configs_match_context_lengths():
    from utils.config import load_yaml

    expected = {"2k": 2048, "4k": 4096, "8k": 8192}
    for name, length in expected.items():
        model = load_yaml(f"configs/model.long_context.{name}.gpu.yaml")
        train = load_yaml(f"configs/pretraining.long_context.{name}.gpu.yaml")
        assert model["max_position"] == length
        assert train["max_sequence_length"] == length
        assert model["rope_original_max_position"] == 1024
        assert model["rope_scaling_type"] in {"ntk", "yarn"}
        assert train["long_context"]["retrieval_validation_required"] is True
        assert train["long_context"]["memory_measurement_required"] is True
