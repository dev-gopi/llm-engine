"""Unit tests for model architecture improvements (Vocab Resize, RoPE, GQA, Checkpointing, Causal Masking)."""

import pytest
import torch

from model.attention import MultiHeadAttention
from model.embedding import TokenEmbedding
from model.gpt import MiniGPT
from model.positional import (
    PositionalEmbedding,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
)


def test_vocab_resize_init_strategies() -> None:
    # Test 'mean' init strategy for TokenEmbedding
    emb = TokenEmbedding(10, 4, padding_idx=0)
    with torch.no_grad():
        emb.embedding.weight[1:].fill_(2.0)
    effective = emb.resize(12, init_strategy="mean")
    assert effective == 12
    torch.testing.assert_close(emb.weight[10], torch.tensor([2.0, 2.0, 2.0, 2.0]))

    # Test 'zero' init strategy for TokenEmbedding
    emb2 = TokenEmbedding(10, 4)
    emb2.resize(14, init_strategy="zero")
    assert torch.equal(emb2.weight[10:], torch.zeros(4, 4))

    # Test invalid init strategy
    with pytest.raises(ValueError, match="init_strategy"):
        emb2.resize(16, init_strategy="invalid_strategy")


def test_minigpt_vocab_resize_with_init_strategy() -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8, lm_head_bias=True)
    new_size = model.resize_token_embeddings(20, init_strategy="zero")
    assert new_size == 20
    assert model.vocab_size == 20
    assert model.head.out_features == 20
    # Head and tok weights should be tied
    assert model.head.weight is model.tok.embedding.weight
    assert torch.equal(model.head.weight[16:], torch.zeros(4, 8))


def test_rotary_positional_embedding() -> None:
    rope = RotaryPositionalEmbedding(dim=8, max_position_embeddings=16, base=10000.0)
    x = torch.randn(2, 2, 6, 8)
    cos, sin = rope(x)
    assert cos.shape == (1, 1, 6, 8)
    assert sin.shape == (1, 1, 6, 8)

    q = torch.randn(2, 2, 6, 8)
    k = torch.randn(2, 2, 6, 8)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_sinusoidal_positional_embedding() -> None:
    pe = SinusoidalPositionalEmbedding(max_pos=16, dim=8)
    tokens = torch.zeros((2, 4), dtype=torch.long)
    output = pe(tokens)
    assert output.shape == (2, 4, 8)

    with pytest.raises(IndexError, match="exceeds max_positions"):
        pe(tokens, position_offset=14)


def test_minigpt_rotary_and_sinusoidal_from_config() -> None:
    # Test rotary model construction & forward pass
    model_rope = MiniGPT.from_config({
        "vocab_size": 32,
        "hidden_size": 16,
        "layers": 2,
        "heads": 4,
        "max_position": 64,
        "position_type": "rotary",
        "rope_base": 10000.0,
    })
    tokens = torch.randint(0, 32, (2, 8))
    logits = model_rope(tokens)
    assert logits.shape == (2, 8, 32)

    # Test sinusoidal model construction & forward pass
    model_sin = MiniGPT.from_config({
        "vocab_size": 32,
        "hidden_size": 16,
        "layers": 2,
        "heads": 4,
        "max_position": 64,
        "position_type": "sinusoidal",
    })
    logits_sin = model_sin(tokens)
    assert logits_sin.shape == (2, 8, 32)


def test_grouped_query_attention() -> None:
    # 8 heads, 2 kv_heads -> GQA ratio 4:1
    attn = MultiHeadAttention(dim=16, heads=8, kv_heads=2)
    assert attn.kv_heads == 2
    assert attn.num_kv_groups == 4

    hidden = torch.randn(2, 6, 16)
    out = attn(hidden)
    assert out.shape == (2, 6, 16)

    # Test KV caching with GQA
    out_cached, (k_cache, v_cache) = attn(hidden, use_cache=True)
    assert k_cache.shape == (2, 2, 6, 2)  # [batch, kv_heads, seq, head_dim]
    assert v_cache.shape == (2, 2, 6, 2)


def test_minigpt_with_gqa() -> None:
    model = MiniGPT.from_config({
        "vocab_size": 32,
        "hidden_size": 16,
        "layers": 2,
        "heads": 8,
        "kv_heads": 2,
        "max_position": 64,
    })
    tokens = torch.randint(0, 32, (2, 8))
    logits = model(tokens)
    assert logits.shape == (2, 8, 32)


def test_minigpt_gradient_checkpointing() -> None:
    model = MiniGPT(vocab_size=32, dim=16, layers=2, heads=4, max_pos=32, gradient_checkpointing=True)
    model.train()
    tokens = torch.randint(0, 32, (2, 8))
    logits = model(tokens)
    loss = logits.sum()
    loss.backward()

    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None

    model.gradient_checkpointing_disable()
    assert not model.gradient_checkpointing


def test_learned_position_interpolation() -> None:
    pos_emb = PositionalEmbedding(max_pos=8, dim=4, interpolate_positions=True)
    tokens = torch.zeros((1, 12), dtype=torch.long)
    # Should not raise IndexError due to interpolate_positions=True
    out = pos_emb(tokens)
    assert out.shape == (1, 12, 4)

    # Test resize with interpolate=True
    pos_emb.resize(16, interpolate=True)
    assert pos_emb.max_positions == 16
    assert pos_emb.weight.shape == (16, 4)
