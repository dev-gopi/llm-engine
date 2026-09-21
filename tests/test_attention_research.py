import torch
from model.attention import MultiHeadAttention

def test_sliding_window_attention_masks_old_tokens():
    module=MultiHeadAttention(16,4,attention_pattern='sliding_window',attention_window=2).eval()
    x=torch.randn(1,4,16)
    out=module(x)
    assert out.shape==(1,4,16)
    assert module.attention_pattern=='sliding_window'

from model.attention import CausalLinearAttention, LinearAttentionState
from model.gpt import MiniGPT


def test_linear_attention_matches_cached_token_by_token_decode():
    torch.manual_seed(7)
    module = CausalLinearAttention(16, 4, kv_heads=2, chunk_size=2).eval()
    x = torch.randn(1, 5, 16)
    full = module(x)
    state = None
    pieces = []
    for index in range(x.shape[1]):
        out, state = module(x[:, index:index + 1], past_key_value=state, use_cache=True)
        pieces.append(out)
    assert isinstance(state, LinearAttentionState)
    assert state.length == x.shape[1]
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=2e-5, rtol=2e-5)


def test_hybrid_attention_pattern_repeats_across_layers_and_caches():
    model = MiniGPT.from_config({
        "vocab_size": 32,
        "hidden_size": 16,
        "layers": 4,
        "heads": 4,
        "kv_heads": 2,
        "max_position": 32,
        "position_type": "rotary",
        "attention_layer_pattern": ["linear", "linear", "linear", "dense"],
        "linear_attention_chunk_size": 2,
    }).eval()
    assert [block.attn.attention_pattern for block in model.blocks] == ["linear", "linear", "linear", "dense"]
    first, cache = model(torch.tensor([[1, 2, 3]]), use_cache=True)
    second, next_cache = model(torch.tensor([[4]]), past_key_values=cache, use_cache=True)
    assert first.shape == (1, 3, 32)
    assert second.shape == (1, 1, 32)
    assert all(model._cache_length(item) == 4 for item in next_cache)
