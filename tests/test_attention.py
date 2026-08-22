import math

import pytest
import torch

from model.attention import MultiHeadAttention


def manual_attention(module: MultiHeadAttention, hidden_states: torch.Tensor) -> torch.Tensor:
    query, key, value = module.project_qkv(hidden_states)
    scores = query @ key.transpose(-2, -1) / math.sqrt(module.head_dim)
    causal = torch.ones(scores.shape[-2:], dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    attended = probabilities @ value
    attended = attended.transpose(1, 2).contiguous().view(*hidden_states.shape)
    return module.out_proj(attended)


def test_qkv_projection_and_output_shapes():
    module = MultiHeadAttention(dim=32, heads=4)
    hidden_states = torch.randn(2, 7, 32)
    query, key, value = module.project_qkv(hidden_states)
    assert query.shape == key.shape == value.shape == (2, 4, 7, 8)
    assert module(hidden_states).shape == hidden_states.shape


def test_matches_manual_causal_reference():
    torch.manual_seed(4)
    module = MultiHeadAttention(dim=16, heads=4, dropout=0.0).eval()
    hidden_states = torch.randn(2, 6, 16)
    expected = manual_attention(module, hidden_states)
    actual = module(hidden_states)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_causal_attention_cannot_see_future_tokens():
    torch.manual_seed(8)
    module = MultiHeadAttention(dim=24, heads=4, causal=True).eval()
    original = torch.randn(1, 6, 24)
    modified = original.clone()
    modified[:, 4:] = torch.randn_like(modified[:, 4:]) * 100
    original_output = module(original)
    modified_output = module(modified)
    torch.testing.assert_close(original_output[:, :4], modified_output[:, :4])
    assert not torch.allclose(original_output[:, 4:], modified_output[:, 4:])


def test_noncausal_attention_can_see_future_tokens():
    torch.manual_seed(9)
    module = MultiHeadAttention(dim=24, heads=4, causal=False).eval()
    original = torch.randn(1, 6, 24)
    modified = original.clone()
    modified[:, -1] += 100
    assert not torch.allclose(module(original)[:, 0], module(modified)[:, 0])


def test_boolean_padding_mask_blocks_keys():
    torch.manual_seed(10)
    module = MultiHeadAttention(dim=16, heads=4, causal=False).eval()
    original = torch.randn(1, 4, 16)
    modified = original.clone()
    modified[:, -1] += 1_000
    mask = torch.tensor([[True, True, True, False]])
    torch.testing.assert_close(module(original, mask)[:, :3], module(modified, mask)[:, :3])


def test_additive_mask_matches_boolean_mask():
    module = MultiHeadAttention(dim=16, heads=4, causal=False).eval()
    hidden_states = torch.randn(2, 4, 16)
    boolean_mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    additive_mask = torch.zeros_like(boolean_mask, dtype=hidden_states.dtype)
    additive_mask.masked_fill_(~boolean_mask, float("-inf"))
    torch.testing.assert_close(
        module(hidden_states, boolean_mask), module(hidden_states, additive_mask)
    )


def test_integer_mask_uses_one_as_allowed():
    module = MultiHeadAttention(dim=16, heads=4, causal=False).eval()
    hidden_states = torch.randn(1, 4, 16)
    integer_mask = torch.tensor([[1, 1, 0, 0]])
    boolean_mask = integer_mask.bool()
    torch.testing.assert_close(
        module(hidden_states, integer_mask), module(hidden_states, boolean_mask)
    )


def test_token_by_token_kv_cache_matches_full_sequence():
    torch.manual_seed(11)
    module = MultiHeadAttention(dim=32, heads=4, causal=True).eval()
    hidden_states = torch.randn(2, 7, 32)
    full_output = module(hidden_states)

    cache = None
    incremental_outputs = []
    for position in range(hidden_states.size(1)):
        output, cache = module(
            hidden_states[:, position : position + 1],
            past_key_value=cache,
            use_cache=True,
        )
        incremental_outputs.append(output)

    incremental_output = torch.cat(incremental_outputs, dim=1)
    torch.testing.assert_close(incremental_output, full_output, rtol=1e-5, atol=1e-6)
    assert cache is not None
    assert cache[0].shape == cache[1].shape == (2, 4, 7, 8)


def test_backward_is_finite():
    module = MultiHeadAttention(dim=32, heads=4)
    hidden_states = torch.randn(2, 8, 32, requires_grad=True)
    module(hidden_states).square().mean().backward()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert all(parameter.grad is not None for parameter in module.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def test_dropout_is_disabled_during_evaluation():
    module = MultiHeadAttention(dim=16, heads=4, dropout=0.5).eval()
    hidden_states = torch.randn(2, 5, 16)
    torch.testing.assert_close(module(hidden_states), module(hidden_states))


def test_from_model_config():
    module = MultiHeadAttention.from_config(
        {
            "hidden_size": 48,
            "heads": 6,
            "attention_dropout": 0.1,
            "attention_bias": False,
            "causal_attention": True,
            "initializer_range": 0.01,
        }
    )
    assert module.dim == 48
    assert module.heads == 6
    assert module.head_dim == 8
    assert module.dropout == 0.1
    assert module.qkv_proj.bias is None
    assert module.causal


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim": 0, "heads": 1},
        {"dim": 16, "heads": 0},
        {"dim": 15, "heads": 4},
        {"dim": 16, "heads": 4, "dropout": 1.0},
        {"dim": 16, "heads": 4, "initializer_range": 0.0},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        MultiHeadAttention(**kwargs)


def test_invalid_input_mask_and_cache():
    module = MultiHeadAttention(dim=16, heads=4)
    with pytest.raises(ValueError, match="shape"):
        module(torch.randn(2, 3, 15))
    with pytest.raises(ValueError, match="2D attention_mask"):
        module(torch.randn(2, 3, 16), torch.ones(2, 2, dtype=torch.bool))
    with pytest.raises(TypeError, match="tuple"):
        module(torch.randn(2, 3, 16), past_key_value=torch.empty(0))
