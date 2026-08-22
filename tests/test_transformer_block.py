import torch
import torch.nn as nn

from model.layer_norm import LayerNorm, RMSNorm
from model.transformer_block import TransformerBlock


def test_pre_norm_block_matches_explicit_residual_equation():
    torch.manual_seed(32)
    block = TransformerBlock(16, 4, residual_scale=0.75).eval()
    hidden_states = torch.randn(2, 6, 16)
    attention_update = block.attn(block.attention_norm(hidden_states))
    after_attention = hidden_states + attention_update * 0.75
    expected = after_attention + block.ffn(block.ffn_norm(after_attention)) * 0.75
    torch.testing.assert_close(block(hidden_states), expected)


def test_zero_sublayers_preserve_exact_residual_stream():
    block = TransformerBlock(16, 4, pre_norm=True).eval()
    for parameter in block.attn.parameters():
        nn.init.zeros_(parameter)
    for parameter in block.ffn.parameters():
        nn.init.zeros_(parameter)
    hidden_states = torch.randn(2, 5, 16)
    torch.testing.assert_close(block(hidden_states), hidden_states, rtol=0, atol=0)


def test_zero_residual_scale_is_identity():
    block = TransformerBlock(16, 4, residual_scale=0.0).eval()
    hidden_states = torch.randn(2, 5, 16)
    torch.testing.assert_close(block(hidden_states), hidden_states, rtol=0, atol=0)


def test_post_norm_mode_matches_explicit_equation():
    block = TransformerBlock(16, 4, pre_norm=False).eval()
    hidden_states = torch.randn(2, 5, 16)
    after_attention = block.attention_norm(hidden_states + block.attn(hidden_states))
    expected = block.ffn_norm(after_attention + block.ffn(after_attention))
    torch.testing.assert_close(block(hidden_states), expected)


def test_residual_dropout_is_disabled_during_evaluation():
    block = TransformerBlock(16, 4, residual_dropout=0.8).eval()
    hidden_states = torch.randn(2, 5, 16)
    torch.testing.assert_close(block(hidden_states), block(hidden_states))


def test_deep_pre_norm_stack_has_finite_gradients():
    torch.manual_seed(33)
    blocks = nn.ModuleList(TransformerBlock(32, 4) for _ in range(8))
    hidden_states = torch.randn(2, 12, 32, requires_grad=True)
    output = hidden_states
    for block in blocks:
        output = block(output)
    output.square().mean().backward()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert hidden_states.grad.norm().item() > 0


def test_block_from_config_builds_all_sublayers():
    block = TransformerBlock.from_config(
        {
            "hidden_size": 32,
            "heads": 4,
            "norm_type": "rms_norm",
            "norm_eps": 1e-6,
            "norm_bias": False,
            "pre_norm": True,
            "residual_dropout": 0.1,
            "residual_scale": 0.5,
            "attention_dropout": 0.2,
            "attention_bias": False,
            "ffn_hidden_size": 80,
            "ffn_multiple_of": 32,
            "ffn_activation": "swiglu",
            "ffn_dropout": 0.3,
            "ffn_bias": False,
            "initializer_range": 0.01,
        }
    )
    assert isinstance(block.attention_norm, RMSNorm)
    assert isinstance(block.ffn_norm, RMSNorm)
    assert block.pre_norm
    assert block.residual_scale == 0.5
    assert block.attn.dropout == 0.2
    assert block.ffn.hidden_dim == 96
    assert block.ffn.activation_name == "swiglu"


def test_default_block_uses_layer_norm():
    block = TransformerBlock(16, 4)
    assert isinstance(block.attention_norm, LayerNorm)
    assert isinstance(block.ffn_norm, LayerNorm)
